// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.  

#include "PxgContext.h"
#include "cudamanager/PxCudaContext.h"
#include "CmNvtx.h"
#include "common/PxProfileZone.h"
#include "PxgIslandContext.h"
#include "PxgSolverCore.h"
#include "PxvSimStats.h"
#include "DyConstraintPrep.h"
#include "PxgArticulationCore.h"
#include "PxgSoftBodyCore.h"
#include "PxgFEMClothCore.h"
#include "DyDeformableSurface.h"
#include "DyDeformableVolume.h"
#include "PxgSimulationCore.h"
#include "PxgPBDParticleSystemCore.h"
#include "DyIslandManager.h"
#include "PxgNarrowphaseCore.h"
#include "PxgCudaBroadPhaseSap.h"
#include "PxsContactManager.h"
#include "PxgPartitionNode.h"
#include "PxgSolverConstraintDesc.h"
#include "PxgKernelWrangler.h"
#include "PxgKernelIndices.h"
#include "CmFlushPool.h"

// PT: TODO: this doesn't compile anymore these days
//#undef PXG_CONTACT_VALIDATION
//#define PXG_CONTACT_VALIDATION	1

namespace physx
{
#if PXG_CONTACT_VALIDATION
#pragma warning(push)
#pragma warning(disable:4100)
	static bool validateContactPairs(PxU32 startIndex, PxU32 endIndex, PxU32* uniqueIds, PxU32* npIds, PxsContactManagerOutputIterator& outputIter,
		PxU8* basePatchPointer, PxU8* baseContactPointer)
	{
		for (PxU32 a = startIndex; a < endIndex; ++a)
		{
			PxU32 uniqueId = uniqueIds[a];
			PxU32 npId = npIds[uniqueId];
			PxsContactManagerOutput& output = outputIter.getContactManagerOutput(npId);

			PxContactPatch* contactPatches = reinterpret_cast<PxContactPatch*>(output.contactPatches);
			PxContact* contacts = reinterpret_cast<PxContact*>(output.contactPoints);

			PX_ASSERT((contactPatches - reinterpret_cast<PxContactPatch*>(basePatchPointer)) < 655360);
			PX_ASSERT((contacts - reinterpret_cast<PxContact*>(baseContactPointer)) < (3145728));

			PX_ASSERT(output.nbPatches != 0);
			PxU32 totalContact = 0;
			for (PxU32 i = 0; i < output.nbPatches; ++i)
			{
				PxContactPatch& patch = contactPatches[i];
				PX_ASSERT(patch.startContactIndex < output.nbContacts);
				PX_ASSERT(patch.normal.isNormalized());
				totalContact += patch.nbContacts;
			}

			for (PxU32 i = 0; i < output.nbContacts; ++i)
			{

				PX_ASSERT(contacts[i].contact.isFinite());
				PX_ASSERT(PxIsFinite(contacts[i].separation));
			}
			PX_ASSERT(totalContact == output.nbContacts);
		}

		return true;
	}

	static bool validateConstraintPairs(PxU32 startIndex, PxU32 endIndex, PxU32* uniqueIds, PxU32* npIds, PxgConstraintPrePrep* constraintPrePrep, PxU32* solverBodyIndices)
	{
		for (PxU32 a = startIndex; a < endIndex; ++a)
		{
			PxU32 uniqueId = uniqueIds[a];
			PxU32 npId = npIds[uniqueId];
			PxgConstraintPrePrep& prePrep = constraintPrePrep[npId];
			PX_ASSERT(prePrep.mNodeIndexA.index() == PX_INVALID_NODE || prePrep.mNodeIndexA.index() < 16000);
			PX_ASSERT(prePrep.mNodeIndexB.index() == PX_INVALID_NODE || prePrep.mNodeIndexB.index() < 16000);

			PX_ASSERT(prePrep.mNodeIndexA.index() == PX_INVALID_NODE || solverBodyIndices[prePrep.mNodeIndexA.index()] < 16000);
			PX_ASSERT(prePrep.mNodeIndexB.index() == PX_INVALID_NODE || solverBodyIndices[prePrep.mNodeIndexB.index()] < 16000);
		}
		return true;
	}

#pragma warning(pop)
#endif

	class PxgBatchArticulationStaticConstraintPrePrepTask : public Cm::Task
	{
		PX_NOCOPY(PxgBatchArticulationStaticConstraintPrePrepTask)
	private:

		PxU32* mStaticContactIndices;
		PxU32* mStaticJointIndices;
		PxU32* mStaticContactCounts;
		PxU32* mStaticJointCounts;
		PxU32* mSelfContactIndices;
		PxU32* mSelfJointIndices;
		PxU32* mSelfContactCounts;
		PxU32* mSelfJointCounts;
		const PxU32 mStartIndex;
		const PxU32 mEndIndex;
		PxNodeIndex* mNodeIndices;
		PxgBodySimManager& mBodyManager;
		const PxU32 mNbArticulations;

	public:

		static const PxU32 NbPerTask = 512;

		PxgBatchArticulationStaticConstraintPrePrepTask(PxU64 context,
			PxU32* staticContactIndices, PxU32* staticJointIndices, PxU32* staticContactCounts, PxU32* staticJointCounts,
			PxU32* selfContactIndices, PxU32* selfJointIndices, PxU32* selfContactCounts, PxU32* selfJointCounts,
			PxU32 startIndex, PxU32 endIndex, PxNodeIndex* nodeIndices, PxgBodySimManager& bodyManager,
			PxU32 nbArticulations) :
			Cm::Task(context), 
			mStaticContactIndices(staticContactIndices), mStaticJointIndices(staticJointIndices), 
			mStaticContactCounts(staticContactCounts), mStaticJointCounts(staticJointCounts),
			mSelfContactIndices(selfContactIndices), mSelfJointIndices(selfJointIndices),
			mSelfContactCounts(selfContactCounts), mSelfJointCounts(selfJointCounts),
			mStartIndex(startIndex), mEndIndex(endIndex),
			mNodeIndices(nodeIndices), mBodyManager(bodyManager), mNbArticulations(nbArticulations)
		{
		}

		virtual const char* getName() const PX_OVERRIDE PX_FINAL
		{
			return "PxgBatchArticulationStaticConstraintPrePrepTask";
		}

		virtual void runInternal() PX_OVERRIDE PX_FINAL
		{
			nvtxRangePush("px:batchArtiStaticPrePrep");
			const PxU32 stride = mNbArticulations;

			//const PxU32 blockCount = (mNbArticulations + 31)/32;

			for (PxU32 i = mStartIndex; i < mEndIndex; ++i)
			{
				const PxU32 nodeIndex = mNodeIndices[i].index();

				PxgStaticConstraints& staticConstraints = mBodyManager.mStaticConstraints[nodeIndex];
				const PxU32 staticContactCount = staticConstraints.mStaticContacts.size();
				PxgStaticConstraint* uniqueIds = staticConstraints.mStaticContacts.begin();

				mStaticContactCounts[i] = staticContactCount;
				for (PxU32 a = 0, offset = i; a < staticContactCount; ++a, offset += stride)
				{
					mStaticContactIndices[offset] = uniqueIds[a].uniqueId;
				}

				const PxU32 staticJointCount = staticConstraints.mStaticJoints.size();
				uniqueIds = staticConstraints.mStaticJoints.begin();

				mStaticJointCounts[i] = staticJointCount;
				for (PxU32 a = 0, offset = i; a < staticJointCount; ++a, offset += stride)
				{
					mStaticJointIndices[offset] = uniqueIds[a].uniqueId;
				}

				const PxU32 articIndex = mBodyManager.mNodeToRemapMap[nodeIndex];

				PxgArticulationSelfConstraints& selfConstraints = mBodyManager.mArticulationSelfConstraints[articIndex];

				const PxU32 selfContactCount = selfConstraints.mSelfContacts.size();
				PxgSelfConstraint* selfIds = selfConstraints.mSelfContacts.begin();

				mSelfContactCounts[i] = selfContactCount;
				for (PxU32 a = 0, offset = i; a < selfContactCount; ++a, offset += stride)
				{
					mSelfContactIndices[offset] = selfIds[a].uniqueId;
				}

				const PxU32 selfJointCount = selfConstraints.mSelfJoints.size();
				selfIds = selfConstraints.mSelfJoints.begin();

				mSelfJointCounts[i] = selfJointCount;
				for (PxU32 a = 0, offset = i; a < selfJointCount; ++a, offset += stride)
				{
					mSelfJointIndices[offset] = selfIds[a].uniqueId;
				}
			}
			nvtxRangePop();
		}
	};

	class PxgBatchRigidStaticConstraintPrePrepTask : public Cm::Task
	{
		PX_NOCOPY(PxgBatchRigidStaticConstraintPrePrepTask)
	private:

		PxU32* mStaticContactIndices;
		PxU32* mStaticJointIndices;
		PxU32* mStaticContactCounts;
		PxU32* mStaticJointCounts;
		const PxU32 mStartIndex;
		const PxU32 mEndIndex;
		PxNodeIndex* mNodeIndices;
		PxgBodySimManager& mBodyManager;
		const PxU32 mNbBodies;

	public:

		static const PxU32 NbPerTask = 256;

		PxgBatchRigidStaticConstraintPrePrepTask(PxU64 context,
			PxU32* staticContactIndices, PxU32* staticJointIndices, PxU32* staticContactCounts, PxU32* staticJointCounts,
			PxU32 startIndex, PxU32 endIndex, PxNodeIndex* nodeIndices, PxgBodySimManager& bodyManager,
			PxU32 nbBodies) :
			Cm::Task(context),
			mStaticContactIndices(staticContactIndices), mStaticJointIndices(staticJointIndices),
			mStaticContactCounts(staticContactCounts), mStaticJointCounts(staticJointCounts),
			mStartIndex(startIndex), mEndIndex(endIndex),
			mNodeIndices(nodeIndices), mBodyManager(bodyManager), mNbBodies(nbBodies)
		{
		}

		virtual const char* getName() const PX_OVERRIDE PX_FINAL
		{
			return "PxgBatchRigidStaticConstraintPrePrepTask";
		}

		virtual void runInternal() PX_OVERRIDE PX_FINAL
		{
			nvtxRangePush("px:batchRigidStaticPrePrep");
			const PxU32 stride = mNbBodies;

			for (PxU32 i = mStartIndex; i < mEndIndex; ++i)
			{
				const PxU32 nodeIndex = mNodeIndices[i].index();

				PxgStaticConstraints& staticConstraints = mBodyManager.mStaticConstraints[nodeIndex];
				const PxU32 staticContactCount = staticConstraints.mStaticContacts.size();
				PxgStaticConstraint* uniqueIds = staticConstraints.mStaticContacts.begin();

				mStaticContactCounts[i] = staticContactCount;
				for (PxU32 a = 0, offset = i; a < staticContactCount; ++a, offset += stride)
				{
					mStaticContactIndices[offset] = uniqueIds[a].uniqueId;
				}

				const PxU32 staticJointCount = staticConstraints.mStaticJoints.size();
				uniqueIds = staticConstraints.mStaticJoints.begin();

				mStaticJointCounts[i] = staticJointCount;
				for (PxU32 a = 0, offset = i; a < staticJointCount; ++a, offset += stride)
				{
					mStaticJointIndices[offset] = uniqueIds[a].uniqueId;
				}
			}
			nvtxRangePop();
		}
	};

	void PxgCpuConstraintPrePrepTask::runInternal()
	{
		nvtxRangePush("px:cpuConstraintPrePrep");
		PX_PROFILE_ZONE("GpuDynamics.PxgCpuJointPrePrepTask", 0);
		PxU32 currentEdgeIndex = 0;

		for (PxU32 a = 0; a < mNumBatches; ++a)
		{
			PxU32 descStride = PxMin(mNumEdges - currentEdgeIndex, PXG_BATCH_SIZE);

			PxgConstraintBatchHeader& batchHeader = mBatchHeaders[a];
			batchHeader.constraintType = PxgSolverConstraintDesc::eCONSTRAINT_1D;
			batchHeader.mDescStride = PxU16(descStride);
			batchHeader.mConstraintBatchIndex = mConstraintBlockStartIndex + a;
			batchHeader.mStartPartitionIndex = mUniqueIdStartIndex + a * PXG_BATCH_SIZE;
			batchHeader.mask = 0xFFFFFFFF; //Unused

#if	PXG_CONTACT_VALIDATION
			validateConstraintPairs(a, a + descStride, mEdgeIds + a, mNpIds, mConstraintPrePrep, mSolverBodyIndices);
#endif

			currentEdgeIndex += descStride;
		}

		for (PxU32 a = 0; a < mNumEdges; ++a)
		{
			mPinnedEdgeIds[mUniqueIdStartIndex + a] = mEdgeIds[a + mStartEdgeIdx];
		}

		//PxMemCopy(mPinnedEdgeIds + mUniqueIdStartIndex, mEdgeIds, sizeof(PxU32) * mNumEdges);
		nvtxRangePop();
	}

	void PxgCpuArtiConstraintPrePrepTask::runInternal()
	{
		nvtxRangePush("px:cpuArtiConstraintPrePrep");
		PX_PROFILE_ZONE("GpuDynamics.PxgCpuArtiJointPrePrepTask", 0);
		PxU32 currentEdgeIndex = 0;
		for (PxU32 a = 0; a < mNumBatches; ++a)
		{
			PxgConstraintBatchHeader& batchHeader = mBatchHeaders[a];
			PxU32 descStride = PxMin(mNumEdges - currentEdgeIndex, PXG_BATCH_SIZE);

			batchHeader.constraintType = PxU16(mIsContact ? PxgSolverConstraintDesc::eARTICULATION_CONTACT : PxgSolverConstraintDesc::eARTICULATION_CONSTRAINT_1D);
			batchHeader.mDescStride = PxU16(descStride);
			batchHeader.mConstraintBatchIndex = mConstraintBlockStartIndex + a;
			batchHeader.mStartPartitionIndex = mUniqueIdStartIndex + a * PXG_BATCH_SIZE;
			batchHeader.mask = 0xFFFFFFFF; //Unused

#if	PXG_CONTACT_VALIDATION
			validateConstraintPairs(a, a + descStride, mEdgeIds + a, mNpIds, mConstraintPrePrep, mSolverBodyIndices);
#endif
			currentEdgeIndex += descStride;
		}

		for (PxU32 a = 0; a < mNumEdges; ++a)
		{
			mPinnedEdgeIds[mUniqueIdStartIndex + a] = mEdgeIds[a + mStartEdgeIdx];
		}

		//PxMemCopy(mPinnedEdgeIds + mUniqueIdStartIndex, mEdgeIds, sizeof(PxU32) * mNumEdges);
		nvtxRangePop();
	}

	void PxgCpuPrepTask::runInternal()
	{
		nvtxRangePush("px:cpuPrepTask");
		mContext.doConstraintPrePrepCommon(mCont);
		nvtxRangePop();
	}

	PxgGpuContext::PxgGpuContext(Cm::FlushPool& flushPool, IG::SimpleIslandManager& islandManager, PxU32 maxNumPartitions, PxU32 maxNumStaticPartitions,
		bool enableStabilization, bool useEnhancedDeterminism, bool solveArticulationContactLast,
		PxReal maxBiasCoefficient, PxvSimStats& simStats, PxgHeapMemoryAllocatorManager* heapMemoryManager, PxReal lengthScale, bool enableDirectGPUAPI, PxU64 contextID, bool isResidualReportingEnabled, bool isTGS) :
		Dy::Context(islandManager, heapMemoryManager->mMappedMemoryAllocators, simStats, enableStabilization,
			useEnhancedDeterminism, solveArticulationContactLast, maxBiasCoefficient, lengthScale, contextID, isResidualReportingEnabled),
		mTotalEdges(0), mTotalPreviousEdges(0),
		mFlushPool(flushPool), 
		mSolvedThisFrame(false),
		mIncrementalPartition(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators), maxNumPartitions, contextID),
		mActiveNodeIndex(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		mSolverBodyPool(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		mBody2WorldPool(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		mLinkAndJointAndRootStateDataPool(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		mArticulationSleepDataPool(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		mInternalResidualPerArticulationVelIter(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		mInternalResidualPerArticulationPosIter(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		m1dConstraintBatchIndices(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		mContactConstraintBatchIndices(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		mArti1dConstraintBatchIndices(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		mArtiContactConstraintBatchIndices(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		mConstraintsPerPartition(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		mArtiConstraintsPerPartition(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		mSolverBodyDataPool(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		mSolverBodySleepDataPool(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		mSolverTxIDataPool(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		mCachedPositionIterations(0), mCachedVelocityIterations(0),
		mArtiStaticContactCounts(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		mArtiStaticJointCounts(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		mArtiStaticContactIndices(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		mArtiStaticJointIndices(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		mArtiSelfContactCounts(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		mArtiSelfJointCounts(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		mArtiSelfContactIndices(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		mArtiSelfJointIndices(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		mRigidStaticContactCounts(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		mRigidStaticJointCounts(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		mRigidStaticContactIndices(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		mRigidStaticJointIndices(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		mNodeIndicesStagingBuffer(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		mIslandIds(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		mIslandStaticTouchCounts(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		mIsTGS(isTGS),
		mIsExternalForcesEveryTgsIterationEnabled(false),
		mEnableDirectGPUAPI(enableDirectGPUAPI),
		mRecomputeArticulationBlockFormat(false),
		mEnforceConstraintWriteBackToHostCopy(false),

		mPreIntegrationTask	(*this),
		mPrepTask			(*this),
		mGpuPrePrepTask		(*this),
		mGpuIntegrationTask	(*this),
		mGpuTask			(*this),
		mPostSolveTask		(*this)
	{
		mGpuArticulationCore = NULL;
		mGpuBp = NULL;
		mGpuNpCore = NULL;
		mGpuSoftBodyCore = NULL;
		mGpuFEMClothCore = NULL;
		mGpuSimulationCore = NULL;
		mGpuSolverCore = NULL;
		mGpuPBDParticleSystemCore = NULL;

		mMaxNumStaticPartitions = maxNumStaticPartitions;
		mStaticContactMappingCount = 0;
		mStaticContactMaxPerArtic = 8;
		mStaticBufAllocSize = 0;
		mStaticContactMapping_d = 0;
		mStaticContactCounts_d = 0;
		mStaticContactIndices_d = 0;
		mStaticNodeArray_d = 0;
		mStaticNpIndexArray_d = 0;
		mStaticSolverConstants_d = 0;
		mStaticPartIndexArray_d = 0;
		mStaticUniqueIdCounter_d = 0;
	}

	PxgGpuContext::~PxgGpuContext()
	{
		mGpuSolverCore->acquireContext();

		PX_DELETE(mPinnedMemoryAllocator);
		PX_DELETE(mContactStreamAllocators[0]);
		PX_DELETE(mContactStreamAllocators[1]);
		PX_DELETE(mPatchStreamAllocators[0]);
		PX_DELETE(mPatchStreamAllocators[1]);
		PX_DELETE(mForceStreamAllocator);
		PX_DELETE(mFrictionPatchStreamAllocator);

		mGpuSolverCore->releaseStreams();

		mGpuSolverCore->releaseContext();

		PX_DELETE(mThresholdStream);
		PX_DELETE(mForceChangedThresholdStream);

		PX_DELETE(mGpuArticulationCore);
		PX_DELETE(mGpuSolverCore);
	}

	PxgSimulationController* PxgGpuContext::getSimulationController()
	{
		return static_cast<PxgSimulationController*>(mSimulationController);
	}

	void PxgGpuContext::setSimulationController(PxsSimulationController* simulationController)
	{
		mSimulationController = simulationController;
	}

	PxgParticleSystemCore* PxgGpuContext::getGpuParticleSystemCore()
	{
		return mGpuPBDParticleSystemCore;
	}
	
	void PxgGpuContext::mergeResults()
	{
		//Flip the current contact stream
		mCurrentContactStream = 1 - mCurrentContactStream;

		if (mIncrementalPartition.getStaticContactsOnly()
			&& mGpuSolverCore->mCompressedContacts.getSize() > 0)
		{
			// Phase B: narrowphase writes directly to solver's device buffer.
			// GPU kernel parameter receives this as a pointer — works for device memory.
			mContactStreamPool.mDataStream = reinterpret_cast<PxU8*>(
				mGpuSolverCore->mCompressedContacts.getDevicePtr());
			mPatchStreamPool.mDataStream = reinterpret_cast<PxU8*>(
				mGpuSolverCore->mCompressedPatches.getDevicePtr());
		}
		else
		{
			mContactStreamPool.mDataStream = mContactStreamAllocators[mCurrentContactStream]->mStart;
			mPatchStreamPool.mDataStream = mPatchStreamAllocators[mCurrentContactStream]->mStart;
		}

		mContactStreamPool.mSharedDataIndex = 0;
		mPatchStreamPool.mSharedDataIndex = 0;
		mForceStreamPool.mSharedDataIndex = 0;
		mFrictionPatchStreamPool.mSharedDataIndex = 0;

		mContactStreamPool.mSharedDataIndexGPU = 0;
		mPatchStreamPool.mSharedDataIndexGPU = 0;
		mForceStreamPool.mSharedDataIndexGPU = 0;
		mFrictionPatchStreamPool.mSharedDataIndexGPU = 0;
	}

	void PxgGpuContext::getDataStreamBase(void*& contactStreamBase, void*& patchStreamBase, void*& forceAndIndiceStreamBase)
	{
		return mGpuSolverCore->getDataStreamBase(contactStreamBase, patchStreamBase, forceAndIndiceStreamBase);
	}

	void PxgGpuContext::overrideAllStreams(CUstream externalStream)
	{
		mGpuSolverCore->overrideStream(externalStream);
		getNarrowphaseCore()->overrideStream(externalStream);
		getArticulationCore()->overrideStream(externalStream);
		getArticulationCore()->setSolverStream(mGpuSolverCore->getStreamRef());
		if (mGpuBp)
			mGpuBp->overrideStream(externalStream);
		// Enable single-stream mode on CudaContext — skips all event/sync/H2D operations
		getNarrowphaseCore()->mCudaContext->setSingleStreamMode(true);
		// Also skip context push/pop in CudaContextManager
		getNarrowphaseCore()->mCudaContextManager->setSingleStreamMode(true);
		// Single-stream mode requires mStaticContactsOnly (articulation-only, no dynamic rigids)
		// This ensures all H2D copies are genuinely redundant.
		PX_ASSERT(mIncrementalPartition.getStaticContactsOnly() &&
		          "Single-stream mode requires mStaticContactsOnly (articulation-only scenes)");
	}

	void PxgGpuContext::setMaxCollisionPairs(PxU32 maxPairs)
	{
		PxgGpuNarrowphaseCore* np = getNarrowphaseCore();
		np->mMaxCollisionPairs = maxPairs;
		if (maxPairs == 0)
			return;

		CUstream stream = np->mStream;
		PxCudaContext* cudaCtx = np->mCudaContext;

		// Allocate dedicated CM count device variable (for future GPU CM lifecycle)
		if (!np->mCmCount_d)
		{
			cuMemAlloc(&np->mCmCount_d, sizeof(PxU32));
			cuMemsetD32Async(np->mCmCount_d, 0, 1, stream);
		}

		// Pre-allocate CM buffers for collision buckets to prevent runtime cuMalloc.
		// allocateCopyOldDataAsync grows the buffer if needed, preserving existing warmup data.
		// If already large enough, this is a no-op.
		const PxU32 buckets[] = { GPU_BUCKET_ID::eConvex, GPU_BUCKET_ID::eConvexPlane };
		for (PxU32 b = 0; b < 2; ++b)
		{
			PxU32 bucketId = buckets[b];
			PxgGpuContactManagers& cms = np->mGpuContactManagers[bucketId]->mContactManagers;
			PxU32 manifoldSize = BUCKET_ManifoldSize[bucketId];

			cms.mContactManagerInputData.allocateCopyOldDataAsync(maxPairs * sizeof(PxgContactManagerInput), cudaCtx, stream, PX_FL);
			cms.mContactManagerOutputData.allocateCopyOldDataAsync(maxPairs * sizeof(PxsContactManagerOutput), cudaCtx, stream, PX_FL);
			if (manifoldSize > 0)
				cms.mPersistentContactManifolds.allocateCopyOldDataAsync(maxPairs * manifoldSize, cudaCtx, stream, PX_FL);
			cms.mCpuContactManagerMapping.allocateCopyOldDataAsync(maxPairs * sizeof(PxsContactManager*), cudaCtx, stream, PX_FL);
			cms.mShapeInteractions.allocateCopyOldDataAsync(maxPairs * sizeof(Sc::ShapeInteraction*), cudaCtx, stream, PX_FL);
			cms.mRestDistances.allocateCopyOldDataAsync(maxPairs * sizeof(PxReal), cudaCtx, stream, PX_FL);
			cms.mTorsionalProperties.allocateCopyOldDataAsync(maxPairs * sizeof(PxsTorsionalFrictionData), cudaCtx, stream, PX_FL);
			cms.mTempRunsumArray.allocateCopyOldDataAsync(2 * maxPairs * sizeof(PxU32), cudaCtx, stream, PX_FL);
			cms.mTempRunsumArray2.allocateCopyOldDataAsync(maxPairs * sizeof(PxU32), cudaCtx, stream, PX_FL);
			cms.mBlockAccumulationArray.allocateCopyOldDataAsync(sizeof(PxU32) * 256, cudaCtx, stream, PX_FL);
			cms.mLostFoundPairsOutputData.allocateCopyOldDataAsync(maxPairs * sizeof(PxsContactManagerOutputCounts), cudaCtx, stream, PX_FL);
			cms.mLostFoundPairsCms.allocateCopyOldDataAsync(maxPairs * sizeof(PxsContactManager*), cudaCtx, stream, PX_FL);
		}
	}

	//this is the pre-prepare code for block format joints loaded from the non-block format joints

	void PxgGpuContext::doConstraintJointBlockPrePrepGPU()
	{
		//DMA the joint pre-prepare data which constructs in CPU(not D6Joint) to GPU

		// AD: This is not needed for direct-GPU API but downstream things are getting really complex and I cannot 
		// figure out which count I need to adjust to avoid crashing.
		//if (!mEnableDirectGPUAPI)
		{
			PxgJointManager& jointManager = getSimulationController()->getJointManager();

			if (jointManager.getCpuNbRigidConstraints() > 0)
			{
				mGpuSolverCore->gpuMemDMAUpJointData(jointManager.getCpuRigidConstraintData(), jointManager.getCpuRigidConstraintRows(), jointManager.getCpuRigidConstraintData().size(), jointManager.getGpuNbRigidConstraints(),
					PxU32(jointManager.mNbCpuRigidConstraintRows));
			}

			if (jointManager.getCpuNbArtiConstraints() > 0)
			{
				mGpuSolverCore->gpuMemDMAUpArtiJointData(jointManager.getCpuArtiConstraintData(), jointManager.getCpuArtiConstraintRows(), jointManager.getCpuArtiConstraintData().size(), jointManager.getGpuNbArtiConstraints(),
					PxU32(jointManager.mNbCpuArtiConstraintRows));
			}
		}

		// maybe this is also not needed if we have direct-GPU?
		mGpuSolverCore->jointConstraintBlockPrePrepParallel(mNumConstraintBatches + mNumRigidStaticConstraintBatches + mNumArticConstraintBatches + mNumArtiStaticConstraintBatches + mNumArtiSelfConstraintBatches);
	}

	void PxgGpuContext::buildAndUploadContactMapping(CUstream stream)
	{
		PX_PROFILE_ZONE("buildAndUploadContactMapping", 0);

		IG::IslandSim& islandSim = mIslandManager.getAccurateIslandSim();
		const IG::CPUExternalData& islandCpu = islandSim.mCpuData;

		// Build nodeToDenseIdx reverse map
		PxgIslandContext& island = mIslandContextPool[0];
		const PxU32 articulationStartIndex = island.mBodyStartIndex + island.mBodyCount;
		PxNodeIndex* nodeIndices = mActiveNodeIndex.begin() + articulationStartIndex;
		PxHashMap<PxU32, PxU32> nodeToDenseIdx;
		for (PxU32 i = 0; i < mArticulationCount; ++i)
			nodeToDenseIdx.insert(nodeIndices[i].index(), i);

		// Count total contact managers across all narrowphase buckets
		// Also record max per-bucket for graph-capturable fixed-size launch
		PxgGpuNarrowphaseCore* npCore = getNarrowphaseCore();
		PxU32 totalCms = 0;
		for (PxU32 bucket = GPU_BUCKET_ID::eConvex; bucket < GPU_BUCKET_ID::eCount; ++bucket)
		{
			if (!npCore->mContactManagers[bucket]) continue;
			PxU32 n = npCore->mContactManagers[bucket]->mContactManagers.mCpuContactManagerMapping.size();
			totalCms += n;
			mMaxCmsPerBucket[bucket] = PxMax(mMaxCmsPerBucket[bucket], n);
		}
		mMaxTotalCms = PxMax(mMaxTotalCms, totalCms);

		// Build mapping on CPU (only used in normal mode / warmup)
		PxArray<ContactArticMapping> mappings;
		mappings.resize(totalCms);
		PxU32 idx = 0;
		for (PxU32 bucket = GPU_BUCKET_ID::eConvex; bucket < GPU_BUCKET_ID::eCount; ++bucket)
		{
			if (!npCore->mContactManagers[bucket]) continue;
			PxgContactManagers& cms = npCore->mContactManagers[bucket]->mContactManagers;
			const PxU32 nbCms = cms.mCpuContactManagerMapping.size();
			for (PxU32 j = 0; j < nbCms; ++j)
			{
				PxsContactManager* cm = cms.mCpuContactManagerMapping[j];
				ContactArticMapping& m = mappings[idx++];
				if (!cm)
				{
					m.denseArticIdx = 0xFFFFFFFF;
					continue;
				}
				const PxcNpWorkUnit& unit = cm->getWorkUnit();
				if (unit.mFlags & PxcNpWorkUnitFlag::eDISABLE_RESPONSE)
				{
					m.denseArticIdx = 0xFFFFFFFF;
					continue;
				}
				m.node0 = islandCpu.getNodeIndex1(unit.mEdgeIndex);
				m.node1 = islandCpu.getNodeIndex2(unit.mEdgeIndex);
				m.npIndex = unit.mNpIndex;
				m.edgeIndex = unit.mEdgeIndex;

				// Determine dense articulation index
				PxNodeIndex articNode;
				bool isStaticArtic = false;
				if (m.node0.isArticulation() && (!m.node1.isValid() || islandSim.getNode(m.node1).isKinematic()))
				{
					articNode = m.node0;
					isStaticArtic = true;
				}
				else if (m.node1.isArticulation() && (!m.node0.isValid() || islandSim.getNode(m.node0).isKinematic()))
				{
					articNode = m.node1;
					isStaticArtic = true;
				}

				if (isStaticArtic)
				{
					const PxPair<const PxU32, PxU32>* entry = nodeToDenseIdx.find(articNode.index());
					m.denseArticIdx = entry ? entry->second : 0xFFFFFFFF;
				}
				else
				{
					m.denseArticIdx = 0xFFFFFFFF;
				}
			}
		}

		mStaticContactMappingCount = idx;
		mStaticContactMaxPerArtic = 8;  // conservative: max contacts per articulation

		// Compute total allocation needed (128-byte aligned sub-ranges for GPU vectorized access)
		#define ALIGN128(x) (((x) + 127u) & ~127u)
		const PxU32 maxTotalContacts = PxMax(idx, 1u);
		const PxU64 totalNeeded =
			ALIGN128(idx * sizeof(ContactArticMapping)) +
			ALIGN128(mArticulationCount * sizeof(PxU32)) +
			ALIGN128(mArticulationCount * mStaticContactMaxPerArtic * sizeof(PxU32)) +
			ALIGN128(maxTotalContacts * sizeof(PartitionNodeData)) +
			ALIGN128(maxTotalContacts * sizeof(PxU32)) +
			ALIGN128(maxTotalContacts * sizeof(PxgSolverConstraintManagerConstants)) +
			ALIGN128(maxTotalContacts * sizeof(PartitionIndexData)) +
			sizeof(PxU32);

		// Allocate single device buffer (lazy, only grow)
		if (mStaticBufAllocSize < totalNeeded)
		{
			freeStaticContactBuffers();
			CUdeviceptr base = 0;
			PxCudaContext* cudaCtx = getNarrowphaseCore()->mCudaContext;
			CUresult res = cudaCtx->memAlloc(&base, totalNeeded);
			if (res != CUDA_SUCCESS)
			{
				PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL,
					"memAlloc for static contact buffers failed! size=%llu\n", (unsigned long long)totalNeeded);
				return;
			}

			// Assign sub-ranges with 128-byte alignment (GPU kernels use vectorized loads)
			CUdeviceptr p = base;
			mStaticContactMapping_d = p; p += ALIGN128(idx * sizeof(ContactArticMapping));
			mStaticContactCounts_d  = p; p += ALIGN128(mArticulationCount * sizeof(PxU32));
			mStaticContactIndices_d = p; p += ALIGN128(mArticulationCount * mStaticContactMaxPerArtic * sizeof(PxU32));
			mStaticNodeArray_d      = p; p += ALIGN128(maxTotalContacts * sizeof(PartitionNodeData));
			mStaticNpIndexArray_d   = p; p += ALIGN128(maxTotalContacts * sizeof(PxU32));
			mStaticSolverConstants_d = p; p += ALIGN128(maxTotalContacts * sizeof(PxgSolverConstraintManagerConstants));
			mStaticPartIndexArray_d = p; p += ALIGN128(maxTotalContacts * sizeof(PartitionIndexData));
			mStaticUniqueIdCounter_d = p;
			mStaticBufAllocSize = totalNeeded;
		}
		#undef ALIGN128

		// Upload mapping
		PxCudaContext* cudaCtx = getNarrowphaseCore()->mCudaContext;
		cudaCtx->memcpyHtoDAsync(mStaticContactMapping_d, mappings.begin(),
			idx * sizeof(ContactArticMapping), stream);

		// Build nodeToDenseIdx lookup table on device (for GPU contact mapping kernel)
		if (mNodeToDenseIdx_d == 0)
		{
			PxU32 maxNodeIdx = 0;
			for (PxU32 i = 0; i < mArticulationCount; ++i)
				maxNodeIdx = PxMax(maxNodeIdx, nodeIndices[i].index());

			mNodeToDenseIdxSize = maxNodeIdx + 1;
			cudaCtx->memAlloc(&mNodeToDenseIdx_d, mNodeToDenseIdxSize * sizeof(PxU32));
			cudaCtx->memsetD32Async(mNodeToDenseIdx_d, 0xFFFFFFFF, mNodeToDenseIdxSize, stream);

			PxArray<PxU32> hostLookup;
			hostLookup.resize(mNodeToDenseIdxSize, 0xFFFFFFFF);
			for (PxU32 i = 0; i < mArticulationCount; ++i)
				hostLookup[nodeIndices[i].index()] = i;
			cudaCtx->memcpyHtoDAsync(mNodeToDenseIdx_d, hostLookup.begin(),
				mNodeToDenseIdxSize * sizeof(PxU32), stream);
		}
	}

	void PxgGpuContext::launchBuildContactMappingGPU(CUstream stream)
	{
		PX_PROFILE_ZONE("launchBuildContactMappingGPU", 0);

		PxgGpuNarrowphaseCore* npCore = getNarrowphaseCore();
		PxCudaContext* cudaCtx = npCore->mCudaContext;

		// nodeToDenseIdx must be built during warmup (buildAndUploadContactMapping)
		PX_ASSERT(mNodeToDenseIdx_d != 0 && "nodeToDenseIdx not initialized — warmup must run first");

		// Use max CM counts from warmup (fixed for graph replay)
		PX_ASSERT(mMaxTotalCms > 0 && "Max CM count not recorded — warmup must run first");

		// Clear valid count
		cudaCtx->memsetD32Async(mStaticUniqueIdCounter_d, 0, 1, stream);

		// Launch one kernel per bucket with max count (graph-safe: fixed grid size)
		CUfunction func = getArticulationCore()->getKernelFunction(PxgKernelIds::BUILD_CONTACT_MAPPING_GPU);
		CUdeviceptr shapeRemapPtr = npCore->mGpuShapesManager.mGpuShapesRemapTableBuffer.getDevicePtr();

		PxU32 cmOffset = 0;
		for (PxU32 bucket = GPU_BUCKET_ID::eConvex; bucket < GPU_BUCKET_ID::eCount; ++bucket)
		{
			PxU32 nCms = mMaxCmsPerBucket[bucket];  // fixed max from warmup
			if (nCms == 0) continue;
			if (!npCore->mContactManagers[bucket]) continue;

			PxgGpuContactManagers& gpuCMs = npCore->getExistingGpuContactManagers(GPU_BUCKET_ID::Enum(bucket));
			CUdeviceptr cmInputPtr = gpuCMs.mContactManagerInputData.getDevicePtr();
			CUdeviceptr nodeToDensePtr = mNodeToDenseIdx_d;
			CUdeviceptr mappingOutPtr = mStaticContactMapping_d + cmOffset * sizeof(ContactArticMapping);
			CUdeviceptr validCountPtr = mStaticUniqueIdCounter_d;

			void* kernelParams[] = {
				&cmInputPtr, &shapeRemapPtr, &nodeToDensePtr,
				&nCms, &mappingOutPtr, &validCountPtr
			};

			const PxU32 blockSize = 256;
			const PxU32 numBlocks = (nCms + blockSize - 1) / blockSize;
			CUresult res = cuLaunchKernel(func, numBlocks, 1, 1, blockSize, 1, 1, 0, stream, kernelParams, NULL);
			if (res != CUDA_SUCCESS)
				PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL,
					"GPU buildContactMappingGPU fail! %d\n", res);

			cmOffset += nCms;
		}

		mStaticContactMappingCount = mMaxTotalCms;
	}

	void PxgGpuContext::launchBuildStaticContactLists(CUstream stream)
	{
		PX_PROFILE_ZONE("launchBuildStaticContactLists", 0);

		PxCudaContext* cudaCtx = getNarrowphaseCore()->mCudaContext;

		// Clear counts and counter
		cudaCtx->memsetD32Async(mStaticContactCounts_d, 0, mArticulationCount, stream);
		cudaCtx->memsetD32Async(mStaticUniqueIdCounter_d, 0, 1, stream);

		if (mStaticContactMappingCount == 0)
			return;

		CUfunction func = getArticulationCore()->getKernelFunction(PxgKernelIds::BUILD_STATIC_CONTACT_LISTS);

		CUdeviceptr mappingPtr = mStaticContactMapping_d;
		PxU32 nCms = mStaticContactMappingCount;
		CUdeviceptr countsPtr = mStaticContactCounts_d;
		CUdeviceptr indicesPtr = mStaticContactIndices_d;
		CUdeviceptr nodePtr = mStaticNodeArray_d;
		CUdeviceptr npIdxPtr = mStaticNpIndexArray_d;
		CUdeviceptr solverPtr = mStaticSolverConstants_d;
		CUdeviceptr partIdxPtr = mStaticPartIndexArray_d;
		PxU32 stride = mArticulationCount;
		PxU32 maxPerArtic = mStaticContactMaxPerArtic;
		CUdeviceptr counterPtr = mStaticUniqueIdCounter_d;

		void* kernelParams[] = {
			&mappingPtr, &nCms, &countsPtr, &indicesPtr,
			&nodePtr, &npIdxPtr, &solverPtr, &partIdxPtr,
			&stride, &maxPerArtic, &counterPtr
		};

		const PxU32 blockSize = 256;
		const PxU32 numBlocks = (nCms + blockSize - 1) / blockSize;

		CUresult res = cuLaunchKernel(func, numBlocks, 1, 1, blockSize, 1, 1, 0, stream, kernelParams, NULL);
		if (res != CUDA_SUCCESS)
			PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL,
				"GPU buildStaticContactListsLaunch fail! %d\n", res);
	}

	void PxgGpuContext::freeStaticContactBuffers()
	{
		if (mStaticContactMapping_d)
		{
			PxCudaContext* cudaCtx = getNarrowphaseCore()->mCudaContext;
			cudaCtx->memFree(mStaticContactMapping_d);
			mStaticContactMapping_d = 0;
			mStaticBufAllocSize = 0;
		}
	}

	void PxgGpuContext::rebuildStaticContactListsDirect(physx::PxBaseTask* continuation,
		const PxU32 articulationConstraintBatchIndex, const PxU32 articulationContactBatchIndex)
	{
		PX_PROFILE_ZONE("rebuildStaticContactListsDirect", 0);
		printf("[rebuildStaticContactLists] ENTER artic=%u\n", mArticulationCount);

		PxgBodySimManager& bodyManager = getSimulationController()->getBodySimManager();

		PxgIslandContext& island = mIslandContextPool[0];
		const PxU32 articulationStartIndex = island.mBodyStartIndex + island.mBodyCount;
		PxNodeIndex* nodeIndices = mActiveNodeIndex.begin() + articulationStartIndex;

		mArtiStaticConstraintBatchOffset = articulationConstraintBatchIndex;
		mArtiStaticContactBatchOffset = articulationContactBatchIndex;

		const PxU32 stride = mArticulationCount;

		// Resize output arrays
		mArtiStaticContactCounts.resize(stride);
		mArtiStaticJointCounts.resize(stride);
		mArtiSelfContactCounts.resize(stride);
		mArtiSelfJointCounts.resize(stride);

		// Clear counts
		PxMemZero(mArtiStaticContactCounts.begin(), stride * sizeof(PxU32));
		PxMemZero(mArtiStaticJointCounts.begin(), stride * sizeof(PxU32));
		PxMemZero(mArtiSelfContactCounts.begin(), stride * sizeof(PxU32));
		PxMemZero(mArtiSelfJointCounts.begin(), stride * sizeof(PxU32));

		// Build nodeIndex → dense articulation index mapping
		PxHashMap<PxU32, PxU32> nodeToDenseIdx;
		for (PxU32 i = 0; i < stride; ++i)
			nodeToDenseIdx.insert(nodeIndices[i].index(), i);

		IG::IslandSim& islandSim = mIslandManager.getAccurateIslandSim();
		const IG::CPUExternalData& islandCpu = islandSim.mCpuData;

		// Count max contacts per articulation (first pass)
		PxU32 totalContacts = 0;
		PxU32 maxPerArtic = 0;

		PxgGpuNarrowphaseCore* npCore = getNarrowphaseCore();
		// Iterate convex-plane bucket (articulation ↔ ground contacts)
		for (PxU32 bucket = GPU_BUCKET_ID::eConvex; bucket < GPU_BUCKET_ID::eCount; ++bucket)
		{
			if (!npCore->mContactManagers[bucket])
				continue;
			PxgContactManagers& cms = npCore->mContactManagers[bucket]->mContactManagers;
			const PxU32 nbCms = cms.mCpuContactManagerMapping.size();
			for (PxU32 j = 0; j < nbCms; ++j)
			{
				PxsContactManager* cm = cms.mCpuContactManagerMapping[j];
				if (!cm) continue;
				const PxcNpWorkUnit& unit = cm->getWorkUnit();
				const PxsContactManagerOutput& output = mOutputIterator.getContactManagerOutput(unit.mNpIndex);
				if (output.nbPatches == 0) continue;
				if (unit.mFlags & PxcNpWorkUnitFlag::eDISABLE_RESPONSE) continue;

				const PxNodeIndex node0 = islandCpu.getNodeIndex1(unit.mEdgeIndex);
				const PxNodeIndex node1 = islandCpu.getNodeIndex2(unit.mEdgeIndex);

				// Determine if this is artic ↔ static
				PxNodeIndex articNode;
				bool isStaticArtic = false;
				if (node0.isArticulation() && (!node1.isValid() || islandSim.getNode(node1).isKinematic()))
				{
					articNode = node0;
					isStaticArtic = true;
				}
				else if (node1.isArticulation() && (!node0.isValid() || islandSim.getNode(node0).isKinematic()))
				{
					articNode = node1;
					isStaticArtic = true;
				}

				if (isStaticArtic)
				{
					const PxPair<const PxU32, PxU32>* entry = nodeToDenseIdx.find(articNode.index());
					if (entry)
						totalContacts += output.nbPatches;
				}
			}
		}

		// Resize index arrays
		maxPerArtic = totalContacts > 0 ? (totalContacts / stride + 4) : 1;
		mArtiStaticContactIndices.resize(maxPerArtic * stride);
		mArtiStaticJointIndices.resize(0);
		mArtiSelfContactIndices.resize(0);
		mArtiSelfJointIndices.resize(0);

		// Ensure partition data arrays are large enough
		mIncrementalPartition.mNpIndexArray.reserve(totalContacts);
		mIncrementalPartition.mNpIndexArray.resizeUninitialized(totalContacts);
		mIncrementalPartition.mPartitionNodeArray.reserve(totalContacts);
		mIncrementalPartition.mPartitionNodeArray.resizeUninitialized(totalContacts);
		mIncrementalPartition.mPartitionIndexArray.reserve(totalContacts);
		mIncrementalPartition.mPartitionIndexArray.resizeUninitialized(totalContacts);
		if (totalContacts > mIncrementalPartition.mSolverConstants.size())
			mIncrementalPartition.mSolverConstants.resize(totalContacts);

		// Second pass: populate arrays
		PxU32 uniqueId = 0;
		for (PxU32 bucket = GPU_BUCKET_ID::eConvex; bucket < GPU_BUCKET_ID::eCount; ++bucket)
		{
			if (!npCore->mContactManagers[bucket])
				continue;
			PxgContactManagers& cms = npCore->mContactManagers[bucket]->mContactManagers;
			const PxU32 nbCms = cms.mCpuContactManagerMapping.size();
			for (PxU32 j = 0; j < nbCms; ++j)
			{
				PxsContactManager* cm = cms.mCpuContactManagerMapping[j];
				if (!cm) continue;
				const PxcNpWorkUnit& unit = cm->getWorkUnit();
				const PxsContactManagerOutput& output = mOutputIterator.getContactManagerOutput(unit.mNpIndex);
				if (output.nbPatches == 0) continue;
				if (unit.mFlags & PxcNpWorkUnitFlag::eDISABLE_RESPONSE) continue;

				const PxNodeIndex node0 = islandCpu.getNodeIndex1(unit.mEdgeIndex);
				const PxNodeIndex node1 = islandCpu.getNodeIndex2(unit.mEdgeIndex);

				PxNodeIndex articNode;
				bool isStaticArtic = false;
				if (node0.isArticulation() && (!node1.isValid() || islandSim.getNode(node1).isKinematic()))
				{
					articNode = node0;
					isStaticArtic = true;
				}
				else if (node1.isArticulation() && (!node0.isValid() || islandSim.getNode(node0).isKinematic()))
				{
					articNode = node1;
					isStaticArtic = true;
				}

				if (!isStaticArtic)
					continue;

				const PxPair<const PxU32, PxU32>* entry = nodeToDenseIdx.find(articNode.index());
				if (!entry)
					continue;
				const PxU32 denseIdx = entry->second;

				for (PxU32 p = 0; p < output.nbPatches; ++p)
				{
					PX_ASSERT(uniqueId < totalContacts);

					// Populate solver data arrays
					mIncrementalPartition.mNpIndexArray[uniqueId] = unit.mNpIndex;

					PartitionNodeData& nd = mIncrementalPartition.mPartitionNodeArray[uniqueId];
					nd.mNodeIndex0 = node0;
					nd.mNodeIndex1 = node1;
					nd.mNextIndex[0] = 0xFFFFFFFF;
					nd.mNextIndex[1] = 0xFFFFFFFF;

					PartitionIndexData& id = mIncrementalPartition.mPartitionIndexArray[uniqueId];
					id.mPatchIndex = PxTo8(p);
					id.mCType = PxU8(IG::Edge::EdgeType::eCONTACT_MANAGER) + 2; // +2 = articulation offset

					mIncrementalPartition.mSolverConstants[uniqueId].mEdgeIndex = unit.mEdgeIndex;

					// Per-articulation contact index
					PxU32 slot = mArtiStaticContactCounts[denseIdx];
					mArtiStaticContactIndices[denseIdx + slot * stride] = uniqueId;
					mArtiStaticContactCounts[denseIdx] = slot + 1;

					uniqueId++;
				}
			}
		}

		// Also update bodySimManager max counts (needed by downstream code)
		bodyManager.mMaxStaticArticContacts = maxPerArtic;
		bodyManager.mMaxStaticArticJoints = 0;
		bodyManager.mMaxSelfArticContacts = 0;
		bodyManager.mMaxSelfArticJoints = 0;
		bodyManager.mTotalStaticArticContacts = uniqueId;

		static int sRebuildCount = 0;
		if (sRebuildCount < 10)
		{
			printf("[rebuildStaticContactLists] frame=%d totalContacts=%u uniqueId=%u stride=%u maxPerArtic=%u\n",
				sRebuildCount, totalContacts, uniqueId, stride, maxPerArtic);
			sRebuildCount++;
		}
	}

	void PxgGpuContext::doStaticArticulationConstraintPrePrep(physx::PxBaseTask* continuation, const PxU32 articulationConstraintBatchIndex, const PxU32 articulationContactBatchIndex)
	{
		if (mIncrementalPartition.getStaticContactsOnly())
		{
			// Phase B: GPU kernel already built per-articulation lists in device buffers.
			return;
		}

		PxgBodySimManager& bodyManager = getSimulationController()->getBodySimManager();

		PxgIslandContext& island = mIslandContextPool[0];

		const PxU32 articulationStartIndex = island.mBodyStartIndex + island.mBodyCount;

		PxNodeIndex* nodeIndices = mActiveNodeIndex.begin() + articulationStartIndex;

		//KS - TODO - revisit this and make it work with batching. Currently, it is disabled!

		mArtiStaticConstraintBatchOffset = articulationConstraintBatchIndex;
		mArtiStaticContactBatchOffset = articulationContactBatchIndex;

		PX_PROFILE_ZONE("Articulation Static constraint", 0);
		mArtiStaticContactCounts.resize(mArticulationCount);
		mArtiStaticJointCounts.resize(mArticulationCount);
		mArtiSelfContactCounts.resize(mArticulationCount);
		mArtiSelfJointCounts.resize(mArticulationCount);

		PxU32 maxArtiStaticContacts = bodyManager.mMaxStaticArticContacts;
		PxU32 maxArtiStaticJoints = bodyManager.mMaxStaticArticJoints;
		PxU32 maxArtiSelfContacts = bodyManager.mMaxSelfArticContacts;
		PxU32 maxArtiSelfJoints = bodyManager.mMaxSelfArticJoints;

		mArtiStaticContactIndices.resize(maxArtiStaticContacts * mArticulationCount);
		mArtiStaticJointIndices.resize(maxArtiStaticJoints * mArticulationCount);

		mArtiSelfContactIndices.resize(maxArtiSelfContacts * mArticulationCount);
		mArtiSelfJointIndices.resize(maxArtiSelfJoints * mArticulationCount);

		for (PxU32 i = 0; i < mArticulationCount; i += PxgBatchArticulationStaticConstraintPrePrepTask::NbPerTask)
		{
			PxU32 endIndex = PxMin(i + PxgBatchArticulationStaticConstraintPrePrepTask::NbPerTask, mArticulationCount);

			PxgBatchArticulationStaticConstraintPrePrepTask* task = PX_PLACEMENT_NEW(mFlushPool.allocate(sizeof(PxgBatchArticulationStaticConstraintPrePrepTask)), PxgBatchArticulationStaticConstraintPrePrepTask)
				(0, mArtiStaticContactIndices.begin(), mArtiStaticJointIndices.begin(), mArtiStaticContactCounts.begin(), mArtiStaticJointCounts.begin(),
					mArtiSelfContactIndices.begin(), mArtiSelfJointIndices.begin(), mArtiSelfContactCounts.begin(), mArtiSelfJointCounts.begin(),
					i, endIndex, nodeIndices, bodyManager, mArticulationCount);

			task->setContinuation(continuation);
			task->removeReference();
		}
	}

	void PxgGpuContext::doStaticRigidConstraintPrePrep(physx::PxBaseTask* continuation)
	{
		PX_PROFILE_ZONE("Rigid Static constraint", 0);
		PxgBodySimManager& bodyManager = getSimulationController()->getBodySimManager();

		PxgIslandContext& island = mIslandContextPool[0];

		const PxU32 bodyStartIndex = island.mBodyStartIndex;

		PxNodeIndex* nodeIndices = mActiveNodeIndex.begin() + bodyStartIndex;

		mRigidStaticContactCounts.resize(mBodyCount);
		mRigidStaticJointCounts.resize(mBodyCount);

		PxU32 maxRigidStaticContacts = bodyManager.mMaxStaticRBContacts;
		PxU32 maxRigidStaticJoints = bodyManager.mMaxStaticRBJoints;

		mRigidStaticContactIndices.resize(maxRigidStaticContacts * mBodyCount);
		mRigidStaticJointIndices.resize(maxRigidStaticJoints * mBodyCount);

		for (PxU32 i = 0; i < mBodyCount; i += PxgBatchArticulationStaticConstraintPrePrepTask::NbPerTask)
		{
			PxU32 endIndex = PxMin(i + PxgBatchArticulationStaticConstraintPrePrepTask::NbPerTask, mBodyCount);

			PxgBatchRigidStaticConstraintPrePrepTask* task = PX_PLACEMENT_NEW(mFlushPool.allocate(sizeof(PxgBatchRigidStaticConstraintPrePrepTask)), PxgBatchRigidStaticConstraintPrePrepTask)
				(0, mRigidStaticContactIndices.begin(), mRigidStaticJointIndices.begin(), mRigidStaticContactCounts.begin(), mRigidStaticJointCounts.begin(),
					i, endIndex, nodeIndices, bodyManager, mBodyCount);

			task->setContinuation(continuation);
			task->removeReference();
		}
	}

	void PxgGpuContext::doConstraintSolveGPU(PxU32 maxNodes, PxBitMapPinned& changedHandleMap)
	{
		/**
		* Things to do in here:
		* (1) Solve on GPU
		* (2) Write-back on GPU
		* (2) Integration on GPU (transforms are now on GPU solver body data so might as well use them)
		*/

		mGpuArticulationCore->syncStream();

		mConstraintPositionIterResidualPoolGpu.resize(mConstraintWriteBackPool.size());

		mGpuSolverCore->solveContactMultiBlockParallel(mIslandContextPool, mNumIslandContextPool,
			mIncrementalPartition.getCombinedSlabMaxNbPartitions(), mConstraintsPerPartition, mArtiConstraintsPerPartition, mGravity, mSolveArticulationContactLast,
			mConstraintPositionIterResidualPoolGpu.begin(), mConstraintPositionIterResidualPoolGpu.size(), &mTotalContactError.mPositionIterationErrorAccumulator,
			mArticulationContactErrorPosIter, mInternalResidualPerArticulationPosIter);
		mContactErrorPosIter = &mTotalContactError.mPositionIterationErrorAccumulator;

		if (mHasForceThresholds)
			mGpuSolverCore->accumulatedForceThresholdStream(maxNodes + 1);

		const PxU32 offset = 1 + mKinematicCount;

		//KS - todo - use separate streams. In addition, read number of threshold streams before DMAing back data
		mGpuSolverCore->gpuMemDMAbackSolverData(mForceStreamPool.mDataStream,
			mForceStreamPool.mDataStreamSize - mForceStreamPool.mSharedDataIndex,
			(PxU32)mForceStreamPool.mSharedDataIndex, (PxU32)mForceStreamPool.mSharedDataIndexGPU,
			mForceChangedThresholdStream->begin(), mIncrementalPartition.hasForceThresholds(),
			mConstraintWriteBackPool.begin(), mConstraintWriteBackPool.size(),
			(!mEnableDirectGPUAPI || getSimulationController()->getEnableOVDCollisionReadback()), mContactErrorVelIter);

		mGpuSolverCore->integrateCoreParallel(offset, mSolverBodyPool.size());

		mGpuArticulationCore->updateBodies(mDt, !mIsTGS, mEnableDirectGPUAPI);

		mSimulationController->update(changedHandleMap);

		if (isResidualReportingEnabled())
			mArticulationContactErrorVelIter.resize(1);

		if (!mEnableDirectGPUAPI || getSimulationController()->getEnableOVDReadback())
		{
			mGpuArticulationCore->gpuMemDMAbackArticulation(mLinkAndJointAndRootStateDataPool, mArticulationSleepDataPool,
				mInternalResidualPerArticulationVelIter, mArticulationContactErrorVelIter);
		}

		mGpuSolverCore->gpuMemDMAbackSolverBodies(reinterpret_cast<float4*>(mSolverBodyPool.begin()), mSolverBodyPool.size(), mBody2WorldPool,
			mSolverBodySleepDataPool, mEnableDirectGPUAPI && (!getSimulationController()->getEnableOVDReadback()));
	}

	class PxgPostSolveWorkerTask : public Cm::Task
	{
		PxNodeIndex* mNodeIndices;
		PxAlignedTransform* mBodyToWorldPool;
		PxgSolverBodySleepData* mSolverBodySleepDataPool;
		float4* mBodyVelocities;
		PxU32 mNbBodies;
		PxU32 mTotalBodies;
		IG::IslandSim* mIslandSim;

	public:

		PxgPostSolveWorkerTask(PxNodeIndex* nodeIndices, PxAlignedTransform* bodyToWorldPool, PxgSolverBodySleepData* solverBodySleepDataPool, float4* bodyVelocities, PxU32 nbBodies, PxU32 totalBodies,
			IG::IslandSim* islandSim) : Cm::Task(0),
			mNodeIndices(nodeIndices), mBodyToWorldPool(bodyToWorldPool), mSolverBodySleepDataPool(solverBodySleepDataPool), mBodyVelocities(bodyVelocities), mNbBodies(nbBodies), mTotalBodies(totalBodies),
			mIslandSim(islandSim)
		{
		}

		virtual void runInternal() PX_OVERRIDE PX_FINAL
		{
			PX_PROFILE_ZONE("GpuDynamics.PxgPostSolveWorkerTask", 0);

			// AD: skip this if we had GPU errors, will lead to asserts down below
			// for signalling reasons we skip outside.
			
			//copy data from PxgSolverBodyData to PxsBodyCore
			for (PxU32 i = 0; i < mNbBodies; i++)
			{
				const PxU32 index = mNodeIndices[i].index();
				//copy integration data

				const PxgSolverBodySleepData& sleepData = mSolverBodySleepDataPool[i];

				PxsRigidBody& originalBody = *getRigidBodyFromIG(*mIslandSim, PxNodeIndex(index));

				PxsBodyCore& bodyCore = originalBody.getCore();

				originalBody.mLastTransform = bodyCore.body2World;
				const PxAlignedTransform& body2World = mBodyToWorldPool[i];
				bodyCore.body2World = body2World.getTransform();
				const float4& linVel = mBodyVelocities[i];
				const float4& angVel = mBodyVelocities[i + mTotalBodies];
				bodyCore.linearVelocity = PxVec3(linVel.x, linVel.y, linVel.z);
				bodyCore.angularVelocity = PxVec3(angVel.x, angVel.y, angVel.z);

				//copy sleep check data
				bodyCore.solverWakeCounter = sleepData.wakeCounter;
				originalBody.mInternalFlags = PxU8(sleepData.internalFlags);

				PX_ASSERT(bodyCore.linearVelocity.isFinite());
				PX_ASSERT(bodyCore.angularVelocity.isFinite());
			}
		}

		virtual const char* getName() const	PX_OVERRIDE PX_FINAL
		{
			return "PxgPostSolveWorkerTask";
		}

	private:
		PX_NOCOPY(PxgPostSolveWorkerTask)
	};


	class PxgPostSolveArticulationTask : public Cm::Task
	{
		PxNodeIndex* mNodeIndices;

		//see PxgArticulationLinkJointRootStateData
		PxU8* mLinkAndJointAndRootStates;
		Dy::ErrorAccumulator* mInternalResidualPerArticulationVelIter;
		Dy::ErrorAccumulator* mInternalResidualPerArticulationPosIter;

		PxgSolverBodySleepData*	mSleepData;
		PxU32 mNbArticulations;
		PxU32 mArticulationStartIndex; //articulation offset in the nodeIndex
		PxU32 mBatchStartIndex;
		IG::SimpleIslandManager* mIslandManager;
		PxU32 mMaxLinks;
		PxU32 mMaxDofs;
		PxReal mDt;
		PxU32 mArticulationCount;

	public:

		PxgPostSolveArticulationTask(PxNodeIndex* nodeIndices, PxU8* linkAndJointAndRootStates, Dy::ErrorAccumulator* internalResidualPerArticulationPosIter, 
			Dy::ErrorAccumulator* internalResidualPerArticulationVelIter, PxgSolverBodySleepData* sleepData, PxU32 nbArticulation,
			PxU32 articulationStartIndex,
			IG::SimpleIslandManager* islandManager, const PxU32 batchStartIndex, const PxU32 maxLinks, const PxU32 maxDofs,
			const PxReal dt, const PxU32 totalArticulationCount) :
			Cm::Task(0), mNodeIndices(nodeIndices),
			mLinkAndJointAndRootStates(linkAndJointAndRootStates),			
			mInternalResidualPerArticulationVelIter(internalResidualPerArticulationVelIter),
			mInternalResidualPerArticulationPosIter(internalResidualPerArticulationPosIter),
			mSleepData(sleepData),
			mNbArticulations(nbArticulation),
			mArticulationStartIndex(articulationStartIndex),
			mBatchStartIndex(batchStartIndex),
			mIslandManager(islandManager),
			mMaxLinks(maxLinks), mMaxDofs(maxDofs),
			mDt(dt), mArticulationCount(totalArticulationCount)
		{
		}

		virtual void runInternal()	PX_OVERRIDE PX_FINAL
		{
			PX_PROFILE_ZONE("GpuDynamics.PxgPostSolveArticulationTask", 0);

			const PxU32 maxLinks = mMaxLinks;
			const PxU32 maxDofs = mMaxDofs;

			//copy data from PxgSolverBodyData to PxsBodyCore
			const PxU32 endIndex = mBatchStartIndex + mNbArticulations;

			IG::IslandSim& sim = mIslandManager->getAccurateIslandSim();

			for (PxU32 a = mBatchStartIndex; a < endIndex; a++)
			{
				const PxU32 ind = a + mArticulationStartIndex;

				PxNodeIndex nodeIndex = mNodeIndices[ind];
				//const PxU32 nodeIndex = mNodeIndices[ind].index();
				//copy integration data
			
				Dy::FeatherstoneArticulation& articulation = *getArticulationFromIG(sim, nodeIndex);
				Dy::ArticulationData& artiData = articulation.getArticulationData();

				articulation.mInternalErrorAccumulatorPosIter = mInternalResidualPerArticulationPosIter[a];
				articulation.mInternalErrorAccumulatorVelIter = mInternalResidualPerArticulationVelIter[a];

				articulation.mContactErrorAccumulatorPosIter = mInternalResidualPerArticulationPosIter[a + mArticulationCount];
				articulation.mContactErrorAccumulatorVelIter = mInternalResidualPerArticulationVelIter[a + mArticulationCount];

				artiData.setDt(mDt);

				const PxU32 numLinks = artiData.getLinkCount();
				const PxU32 numDofs = artiData.getDofs();

				//Get the address of the buffer holding the state data for the current articulation.
				PxU8* singleArticulationStateBuffer = 
					PxgArticulationLinkJointRootStateData::getArticulationStateDataBuffer(
						mLinkAndJointAndRootStates,
						maxLinks, maxDofs, a);

				//Decompose the buffer into its sub-arrays.
				PxTransform* sBody2Worlds = NULL;
				Cm::UnAlignedSpatialVector* sLinkVelocities = NULL;
				Cm::UnAlignedSpatialVector* sLinkAccelerations = NULL;
				Cm::UnAlignedSpatialVector* sLinkIncomingJointForces = NULL;
				PxReal* sJointPositions = NULL;
				PxReal* sJointVelocities = NULL;
				PxReal* sJointAccels = NULL;
				Cm::UnAlignedSpatialVector* sRootPreVel = NULL;
				PxgArticulationLinkJointRootStateData::decomposeArticulationStateDataBuffer(
					singleArticulationStateBuffer,
					numLinks, numDofs, 
					sBody2Worlds, sLinkVelocities, sLinkAccelerations, sLinkIncomingJointForces,
					sJointPositions, sJointVelocities, sJointAccels,
					sRootPreVel);
				
				Dy::ArticulationCore* core = articulation.getCore();
				core->wakeCounter = mSleepData[a].wakeCounter;

				if (mSleepData[a].internalFlags & PxsRigidBody::eACTIVATE_THIS_FRAME)
				{
					mIslandManager->getAccurateIslandSim().activateNode_ForGPUSolver(nodeIndex);
					mIslandManager->getSpeculativeIslandSim().activateNode_ForGPUSolver(nodeIndex);
				}
				else if (mSleepData[a].internalFlags & PxsRigidBody::eDEACTIVATE_THIS_FRAME)
				{
					mIslandManager->getAccurateIslandSim().deactivateNode_ForGPUSolver(nodeIndex);
					mIslandManager->getSpeculativeIslandSim().deactivateNode_ForGPUSolver(nodeIndex);
				}

				Dy::ArticulationLink* links = artiData.getLinks();
				Cm::SpatialVectorF* linkVelocities = artiData.getMotionVelocities();
				Cm::SpatialVectorF* linkAccelerations = artiData.getMotionAccelerations();
				Cm::SpatialVectorF* linkIncomingJointForces = artiData.getLinkIncomingJointForces();
				for (PxU32 i = 0; i < numLinks; ++i)
				{
					Dy::ArticulationLink& link = links[i];
					PX_ASSERT(sBody2Worlds[i].isValid());

					link.bodyCore->body2World = sBody2Worlds[i];
					link.bodyCore->angularVelocity = sLinkVelocities[i].top;
					link.bodyCore->linearVelocity = sLinkVelocities[i].bottom;

					linkVelocities[i].top = sLinkVelocities[i].top;
					linkVelocities[i].bottom = sLinkVelocities[i].bottom;

					linkAccelerations[i].top = sLinkAccelerations[i].top;
					linkAccelerations[i].bottom = sLinkAccelerations[i].bottom;

					linkIncomingJointForces[i].top = sLinkIncomingJointForces[i].top;
					linkIncomingJointForces[i].bottom = sLinkIncomingJointForces[i].bottom;
				}
				linkIncomingJointForces[0].top = PxVec3(PxZero);
				linkIncomingJointForces[0].bottom = PxVec3(PxZero);

				PxReal* jointPositions = artiData.getJointPositions();
				PxReal* jointVelocities = artiData.getJointVelocities();
				PxReal* jointAccelerations = artiData.getJointAccelerations();
				for (PxU32 i = 0; i < numDofs; ++i)
				{
					jointPositions[i] = sJointPositions[i];
					jointVelocities[i] = sJointVelocities[i];
					jointAccelerations[i] = sJointAccels[i];
				}

				artiData.setRootPreMotionVelocity(*sRootPreVel);
			}
		}

		virtual const char* getName() const	PX_OVERRIDE PX_FINAL
		{
			return "PxgPostSolveArticulationTask";
		}

	private:
		PX_NOCOPY(PxgPostSolveArticulationTask)
	};

	void PxgGpuContext::processPatches(	Cm::FlushPool& flushPool, PxBaseTask* continuation,
										PxsContactManager** lostFoundPatchManagers, PxU32 nbLostFoundPatchManagers, PxsContactManagerOutputCounts* outCounts)
	{
		mIncrementalPartition.processLostFoundPatches(	flushPool, continuation, mIslandManager.getAccurateIslandSim(),
														getSimulationController()->getBodySimManager(), getSimulationController()->getJointManager(),
														lostFoundPatchManagers, nbLostFoundPatchManagers, outCounts);
	}

	void PxgGpuContext::doPostSolveTask(physx::PxBaseTask* continuation)
	{
		if (!mSolvedThisFrame)
			return;

		// AD: sneaky, but apparently only narrowphasecore has that member public.
		if (getNarrowphaseCore()->mCudaContext->isInAbortMode())
			return;

		const PxU32 numParticleCores = mGpuParticleSystemCores.size();
		for (PxU32 i = 0; i < numParticleCores; ++i)
		{
			PxgParticleSystemCore* core = mGpuParticleSystemCores[i];
			const PxReal eps = 0.f;// mLengthScale * 1e-4f;
			core->integrateSystems(mDt, eps*eps);
			core->onPostSolve(); // call the callback.
		}

		PxU32 nbThresholdElems = 0;
		mGpuSolverCore->syncDmaBack(nbThresholdElems);
		mForceChangedThresholdStream->forceSize_Unsafe(nbThresholdElems);

		if (!mEnableDirectGPUAPI || getSimulationController()->getEnableOVDReadback())
		{
			//TODO - multi-thread this!
			const PxU32 offset = 1 + mKinematicCount;

			PxPinnedArray<PxgSolverBody>& solverBodyIter = mSolverBodyPool;

			float4* bodyVelocities = reinterpret_cast<float4*>(solverBodyIter.begin());
			PxAlignedTransform* body2Worlds = mBody2WorldPool.begin();
			PxNodeIndex* nodeIndices = mActiveNodeIndex.begin();
			const PxU32 totalNumBodies = mSolverBodyPool.size();

			const PxU32 batchSize = 512;

			IG::IslandSim* accurateIslandSim = &mIslandManager.getAccurateIslandSim();

			//write back the data to PxsBodyCore
			for (PxU32 i = offset; i < totalNumBodies; i += batchSize)
			{
				PxgSolverBodySleepData* sleepData = &mSolverBodySleepDataPool[i];

				PxgPostSolveWorkerTask* task = PX_PLACEMENT_NEW(mFlushPool.allocate(sizeof(PxgPostSolveWorkerTask)), PxgPostSolveWorkerTask)(nodeIndices + i, body2Worlds + i, sleepData, bodyVelocities + i,
					PxMin(batchSize, totalNumBodies - i), totalNumBodies, accurateIslandSim);

				task->setContinuation(continuation);
				task->removeReference();
			}

			const PxU32 maxLinks = getSimulationController()->getSimulationCore()->getMaxArticulationLinks();
			const PxU32 maxDofs = getSimulationController()->getSimulationCore()->getMaxArticulationDofs();
			const PxU32 articulationBatchSize = PxMax(64u, (mArticulationCount + 127u) / 128u);
			for (PxU32 i = 0; i < mArticulationCount; i += articulationBatchSize)
			{
				PxgPostSolveArticulationTask* task = PX_PLACEMENT_NEW(mFlushPool.allocate(sizeof(PxgPostSolveArticulationTask)), PxgPostSolveArticulationTask)(nodeIndices,
					mLinkAndJointAndRootStateDataPool.begin(), mInternalResidualPerArticulationPosIter.begin(), mInternalResidualPerArticulationVelIter.begin(),
					mArticulationSleepDataPool.begin(), PxMin(articulationBatchSize, mArticulationCount - i), mArticulationStartIndex, &mIslandManager, i,
					maxLinks, maxDofs, mDt, mArticulationCount);

				task->setContinuation(continuation);
				task->removeReference();
			}
		}

		mGpuSolverCore->acquireContext();
		for (PxU32 i = 0; i < numParticleCores; ++i)
		{
			PxgParticleSystemCore* core = mGpuParticleSystemCores[i];
			
			cuStreamQuery(core->getFinalizeStream()); //Flush particle work
		}

		mGpuSolverCore->releaseContext();
	}

void PxgGpuContext::simulateLean()
{
	// Lean solver: direct GPU kernel launches for Phase B (artic-only, static contacts).
	// Called from update() in lean mode, after buffer setup and allocateSolverBodyBuffers.
	// All batch counts, partition data, and island context retain values from warmup.
	// Context is already acquired by update().

	PX_PROFILE_ZONE("PxgGpuContext.simulateLean", 0);

	CUstream solverStream = mGpuSolverCore->getStream();
	PxCudaContext* cudaCtx = getNarrowphaseCore()->mCudaContext;

	// 1. Phase B contact mapping: map NP contact output to solver format
	launchBuildContactMappingGPU(solverStream);
	launchBuildStaticContactLists(solverStream);

	// 2. Patch prep descriptor with actual contact count from GPU kernel output
	{
		CUdeviceptr prepDescd = mGpuSolverCore->getPrepDescDeviceptr();
		if (prepDescd)
		{
			cudaCtx->memcpyDtoDAsync(
				prepDescd + offsetof(PxgConstraintPrepareDesc, numArtiStaticContactBatches),
				mStaticUniqueIdCounter_d, sizeof(PxU32), solverStream);
			cudaCtx->memcpyDtoDAsync(
				prepDescd + offsetof(PxgConstraintPrepareDesc, totalCurrentEdges),
				mStaticUniqueIdCounter_d, sizeof(PxU32), solverStream);
		}
	}

	// 3-5. Integration + articulation + internal constraints
	doPreIntegrationGPU();
	{ CUresult r = cuStreamSynchronize(solverStream); if(r) { fprintf(stderr, "[lean] preIntegration: %d\n", (int)r); return; } }
	doArticulationGPU();
	{ CUresult r = cuStreamSynchronize(solverStream); if(r) { fprintf(stderr, "[lean] articulation: %d\n", (int)r); return; } }

	if (mIsTGS)
	{
		const PxReal stepDt = mDt / PxReal(mIslandContextPool->mNumPositionIterations);
		mGpuArticulationCore->setupInternalConstraints(mArticulationCount, stepDt, mDt, 1.0f / stepDt, true);
		{ CUresult r = cuStreamSynchronize(solverStream); if(r) { fprintf(stderr, "[lean] internalConstraints: %d\n", (int)r); return; } }
	}

	// 6. Joint block pre-prep
	doConstraintJointBlockPrePrepGPU();
	{ CUresult r = cuStreamSynchronize(solverStream); if(r) { fprintf(stderr, "[lean] jointBlockPrePrep: %d\n", (int)r); return; } }

	// 6.5. Sync articulation stream, then build batch headers for static contacts
	// (normally done in doConstraintPrePrepGPUCommon)
	mGpuArticulationCore->syncStream();
	mGpuArticulationCore->createStaticContactAndConstraintsBatch(mArticulationCount);
	{ CUresult r = cuStreamSynchronize(solverStream); if(r) { fprintf(stderr, "[lean] createStaticBatch: %d\n", (int)r); return; } }

	// 6.6. The device-side preprep descriptor retains correct values from warmup's
	// H2D transfer (Phase B pointers, batch counts, etc.). GPU kernels
	// (createStaticContactAndConstraintsBatch) update batch counts directly on device.
	// DO NOT re-upload from host — that would overwrite GPU-written values with stale
	// host data and cause error 700 in the preprep kernel.

	// 6.7. constraintPrePrepParallel: rigid static batch setup + contact block pre-prep
	{
		const PxU32 totalBatches = mNumConstraintBatches + mNumRigidStaticConstraintBatches + mNumArticConstraintBatches + mNumArtiStaticConstraintBatches + mNumArtiSelfConstraintBatches;
		fprintf(stderr, "[lean] batches: rigid=%u rigidStatic=%u artic=%u articStatic=%u articSelf=%u total=%u bodyCount=%u\n",
			mNumConstraintBatches, mNumRigidStaticConstraintBatches, mNumArticConstraintBatches,
			mNumArtiStaticConstraintBatches, mNumArtiSelfConstraintBatches, totalBatches, mIslandContextPool->mBodyCount);
		mGpuSolverCore->constraintPrePrepParallel(totalBatches, 0, mIslandContextPool->mBodyCount);
	}
	{ CUresult r = cuStreamSynchronize(solverStream); if(r) { fprintf(stderr, "[lean] constraintPrePrep: %d\n", (int)r); return; } }

	// 7-9. Constraint prep, solve, merge
	doConstraintPrepGPU();
	{ CUresult r = cuStreamSynchronize(solverStream); if(r) { fprintf(stderr, "[lean] constraintPrep: %d\n", (int)r); return; } }

	// Populate partition arrays from incremental partition (normally done in doConstraintPrePrepGPUCommon)
	// In lean mode, doConstraintPrePrepGPUCommon is skipped, so these must be set here.
	{
		const PxU32 nbCombinedSlabPartitions = mIncrementalPartition.getCombinedSlabNbPartitions();
		mConstraintsPerPartition.forceSize_Unsafe(0);
		mArtiConstraintsPerPartition.forceSize_Unsafe(0);
		for (PxU32 a = 0; a < nbCombinedSlabPartitions; ++a)
		{
			mConstraintsPerPartition.pushBack(mIncrementalPartition.getCSlabAccumulatedPartitionCount(a));
			mArtiConstraintsPerPartition.pushBack(mIncrementalPartition.getCSlabAccumulatedArtiPartitionCount(a));
		}
		mIslandContextPool->mStartPartitionIndex = 0;
		mIslandContextPool->mNumPartitions = nbCombinedSlabPartitions;
		mIslandContextPool->mBatchStartIndex = 0;
		mIslandContextPool->mBatchCount = mIncrementalPartition.getNbConstraintBatches() + mIncrementalPartition.getNbContactBatches();
		mIslandContextPool->mArtiBatchStartIndex = 0;
		mIslandContextPool->mArtiBatchCount = mIncrementalPartition.getNbArtiConstraintBatches() + mIncrementalPartition.getNbArtiContactBatches();

		mIslandContextPool->mStaticArtiBatchCount = mNumArtiStaticConstraintBatches;
		mIslandContextPool->mSelfArtiBatchCount = mNumArtiSelfConstraintBatches;
		mIslandContextPool->mStaticRigidBatchCount = mNumRigidStaticConstraintBatches;
	}

	PxBitMapPinned emptyChangedHandleMap;
	doConstraintSolveGPU(mArticulationCount, emptyChangedHandleMap);
	{ CUresult r = cuStreamSynchronize(solverStream); if(r) { fprintf(stderr, "[lean] constraintSolve: %d\n", (int)r); return; } }

	mergeResults();
	{ CUresult r = cuStreamSynchronize(solverStream); if(r) { fprintf(stderr, "[lean] mergeResults: %d\n", (int)r); return; } }
}

static void copyToSolverBodyStaticAndKinematic(PxgSolverBodyData& data, PxgSolverTxIData& txIData, const PxsBodyCore& core, PxNodeIndex nodeIndex)
	{
		// PT: not needed for statics/kinematics
//		if(core.disableGravity)
//			sleepData.internalFlags |= PxsRigidBody::eDISABLE_GRAVITY_GPU;

		//This data has been moved to pxgbodysim
		//data.inverseInertia = make_float4(core.inverseInertia.x, core.inverseInertia.y, core.inverseInertia.z, 0.f);
		//PxU32 islandNodeIndex = nodeIndex << 2;

		////Enable CCD...
		//if (core.mFlags & PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD)
		//	islandNodeIndex |= 1;
		//if (originalBody.mInternalFlags & PxsRigidBody::eHAS_SURFACE_VELOCITY)
		//	islandNodeIndex |= 2;

		data.islandNodeIndex = nodeIndex;

		// Copy simple properties
		data.initialLinVel = core.linearVelocity;
		data.initialAngVel = core.angularVelocity;

		txIData.sqrtInvInertia = PxMat33(PxZero);
		txIData.deltaBody2World = PxTransform(PxIdentity);

		PX_ASSERT(core.linearVelocity.isFinite());
		PX_ASSERT(core.angularVelocity.isFinite());

		data.invMass = core.inverseMass;
		data.penBiasClamp = core.maxPenBias;
		//data.writeIndex = PxgSolverBody::InvalidHandle;

		data.reportThreshold = core.contactReportThreshold;
		data.maxImpulse = core.maxContactImpulse;
		data.offsetSlop = 0.0f;
		data.body2World = PxAlignedTransform(core.body2World.p.x, core.body2World.p.y, core.body2World.p.z,
			PxAlignedQuat(core.body2World.q.x, core.body2World.q.y, core.body2World.q.z, core.body2World.q.w));

		data.flags = PxRigidBodyFlag::eKINEMATIC;
	}

	static void atomArticulationIntegration(const PxU32 numArticulations,
		const PxNodeIndex* const PX_RESTRICT islandNodes,
		IG::SimpleIslandManager& islandManager,
		PxI32* maxPosIters, PxI32* maxVelIters)
	{
		PxU32 localMaxPosIter = 0, localMaxVelIter = 0;
		for (PxU32 a = 0; a < numArticulations; ++a)
		{
			const PxNodeIndex nodeId = islandNodes[a];
			//const PxU32 nodeIndex = nodeId.index();
	
			Dy::FeatherstoneArticulation* artic = getArticulationFromIG(islandManager.getAccurateIslandSim(), nodeId);

			const PxU16 iterCount = artic->getIterationCounts();

			localMaxPosIter = PxMax<PxU32>(PxU32(iterCount & 0xff), localMaxPosIter);
			localMaxVelIter = PxMax<PxU32>(PxU32(iterCount >> 8), localMaxVelIter);
		}

		PxAtomicMax(maxPosIters, (PxI32)localMaxPosIter);
		PxAtomicMax(maxVelIters, (PxI32)localMaxVelIter);
	}

	class PxgSetupKinematicTask : public Cm::Task
	{
		const PxNodeIndex* const PX_RESTRICT	mKinematicNodes;
		PxNodeIndex*							mActiveNodeIndex;		//copy island node index into this list
		const PxU32								mNumBodies;
		IG::SimpleIslandManager&				mIslandManager;
		PxU32									mSolverBodyStartIndex;

		PxgSolverBodyData*						mSolverBodyDataPool;
		PxgSolverBodySleepData*					mSolverBodySleepDataPool;
		PxgSolverTxIData*						mSolverTxIData;

		PX_NOCOPY(PxgSetupKinematicTask)

	public:

		PxgSetupKinematicTask(const PxNodeIndex* const PX_RESTRICT kinematicNodes, PxNodeIndex*	activeNodeIndex, const PxU32 numBodies,
			IG::SimpleIslandManager& islandManager, PxU32 solverBodyStartIndex, PxgSolverBodyData* solverBodyDataPool,
			PxgSolverBodySleepData* solverBodySleepDataPool, PxgSolverTxIData* txIData) : Cm::Task(0), mKinematicNodes(kinematicNodes), mActiveNodeIndex(activeNodeIndex), mNumBodies(numBodies),
			mIslandManager(islandManager), mSolverBodyStartIndex(solverBodyStartIndex), mSolverBodyDataPool(solverBodyDataPool),
			mSolverBodySleepDataPool(solverBodySleepDataPool), mSolverTxIData(txIData)
		{
		}

		virtual void runInternal()	PX_OVERRIDE PX_FINAL
		{
			IG::IslandSim& islandSim = mIslandManager.getAccurateIslandSim();

			//Set up solver bodies for any kinematic bodies
			for (PxU32 i = 0; i < mNumBodies; i++)
			{
				PxsRigidBody& rigidBody = *getRigidBodyFromIG(islandSim, mKinematicNodes[i]);
				const PxsBodyCore& core = rigidBody.getCore();
				copyToSolverBodyStaticAndKinematic(mSolverBodyDataPool[i], mSolverTxIData[i], core, mKinematicNodes[i]);
				//mActiveNodeIndex[mSolverBodyStartIndex + i] = mKinematicNodes[i];
				rigidBody.saveLastCCDTransform();
			}
		}

		virtual const char* getName() const	PX_OVERRIDE PX_FINAL
		{
			return "PxgKinematicSetupTask";
		}
	};

	class PxgAtomIntegrationTask : public Cm::Task
	{
		const PxNodeIndex* const PX_RESTRICT	mIslandNodes;
		const PxU32								mNumBodies;
		PxI32*									mMaxPosIters;
		PxI32*									mMaxVelIters;
		IG::SimpleIslandManager&				mIslandManager;

		PX_NOCOPY(PxgAtomIntegrationTask)

	public:

		PxgAtomIntegrationTask(const PxNodeIndex* const PX_RESTRICT islandNodes, const PxU32 numBodies, PxI32* PX_RESTRICT maxPosIters, PxI32* PX_RESTRICT maxVelIters,
			IG::SimpleIslandManager& islandManager) : Cm::Task(0),
			mIslandNodes(islandNodes),
			mNumBodies(numBodies), mMaxPosIters(maxPosIters), mMaxVelIters(maxVelIters),
			mIslandManager(islandManager)
		{
		}

		virtual void runInternal()	PX_OVERRIDE PX_FINAL
		{
			PX_PROFILE_ZONE("GpuDynamics.PxgIntegrateTask", 0);
			PxI32 localPosIters = 0; PxI32 localVelIters = 0;
			IG::IslandSim& sim = mIslandManager.getAccurateIslandSim();
			for (PxU32 i = 0; i < mNumBodies; ++i)
			{
				const PxNodeIndex nodeId = mIslandNodes[i];
				//activeNodeIndex[startIndex] = nodeId;
				PxsRigidBody& rigidBody = *getRigidBodyFromIG(sim, nodeId);

				localPosIters = PxMax<PxI32>(PxI32(rigidBody.mSolverIterationCounts & 0xff), localPosIters);
				localVelIters = PxMax<PxI32>(PxI32(rigidBody.mSolverIterationCounts >> 8), localVelIters);
			}

			PxAtomicMax(mMaxPosIters, localPosIters);
			PxAtomicMax(mMaxVelIters, localVelIters);
		}

		virtual const char* getName() const	PX_OVERRIDE PX_FINAL
		{
			return "PxgIntegrateTask";
		}
	};

	class PxgArticulationAtomIntegrationTask : public Cm::Task
	{
		const PxNodeIndex* const PX_RESTRICT	mIslandNodes;

		const PxU32								mNumArticulations;

		PxI32*									mMaxPosIters;
		PxI32*									mMaxVelIters;
		IG::SimpleIslandManager&				mIslandManager;

		PX_NOCOPY(PxgArticulationAtomIntegrationTask)

	public:

		PxgArticulationAtomIntegrationTask(
			const PxNodeIndex* const PX_RESTRICT islandNodes,
			const PxU32 numArticulations, PxI32* maxPosIters,
			PxI32* maxVelIters,
			IG::SimpleIslandManager& islandManager
		) :
			Cm::Task(0), mIslandNodes(islandNodes),
			mNumArticulations(numArticulations),
			mMaxPosIters(maxPosIters),
			mMaxVelIters(maxVelIters),
			mIslandManager(islandManager)
		{
		}

		virtual void runInternal()	PX_OVERRIDE PX_FINAL
		{
			PX_PROFILE_ZONE("GpuDynamics.PxgArticulationAtomIntegrationTask", 0);
			atomArticulationIntegration(mNumArticulations, mIslandNodes,
				mIslandManager, mMaxPosIters, mMaxVelIters);
		}

		virtual const char* getName() const	PX_OVERRIDE PX_FINAL
		{
			return "PxgArticulationAtomIntegrationTask";
		}
	};

	void PxgGpuContext::doPreIntegrationTaskCommon(physx::PxBaseTask* continuation)
	{
		// AD: this task currently assumes we only have 1 solver island. If there is a variable amount of islands,
		// the dependency chain needs to be fixed, because this task runs in parallel to allocating and setting
		// the members of mIslandContextPool. (see Pxg(TGS)DynamicsContext::update()).

		mNumContactBatches = 0;
		mNum1dConstraintBatches = 0;
		mNumArtiContactBatches = 0;
		mNumArti1dConstraintBatches = 0;

		mArtiStaticConstraintBatchOffset = 0;
		mArtiStaticContactBatchOffset = 0;

		const IG::IslandSim& islandSim = mIslandManager.getAccurateIslandSim();

		const PxU32 workerCount = PxMax(1u, continuation->getTaskManager()->getCpuDispatcher()->getWorkerCount());

		const PxU32 atomBatchSize = PxMax(256u, PxMin(1024u, (mBodyCount + workerCount - 1) / workerCount));

		const PxNodeIndex* const PX_RESTRICT nodeIndices = islandSim.getActiveNodes(IG::Node::eRIGID_BODY_TYPE);

		mGpuSolverCore->acquireContext();

		const PxNodeIndex* const PX_RESTRICT articulationNodeIndices = islandSim.getActiveNodes(IG::Node::eARTICULATION_TYPE);

		//Because we need to put the articulation active node index into the same list as mActiveNodeIndex, so we need to make sure
		//articulation active node index start in the right place. In the active node index list, we start with static + kinematic +
		//active rigid bodies + active articulations
		//const PxU32 articulationStartIndex = island.mBodyStartIndex + island.mBodyCount;

		if (isStateDirty())
		{
			mCachedPositionIterations = 0;
			mCachedVelocityIterations = 0;

			//Loop through and fill in properties from all the rigid bodies...
			for (PxU32 a = 0; a < mBodyCount; a += atomBatchSize)
			{
				PxgAtomIntegrationTask* task = static_cast<PxgAtomIntegrationTask*>(mFlushPool.allocate(sizeof(PxgAtomIntegrationTask)));

				task = PX_PLACEMENT_NEW(task, PxgAtomIntegrationTask)(nodeIndices + a, PxMin(atomBatchSize, mBodyCount - a), &mCachedPositionIterations,
					&mCachedVelocityIterations, mIslandManager);

				task->setContinuation(continuation);
				task->removeReference();
			}

			setStateDirty(false);

			const PxU32 articulationBatchSize = 1024u;

			for (PxU32 a = 0; a < mArticulationCount; a += articulationBatchSize)
			{
				PxgArticulationAtomIntegrationTask* task = static_cast<PxgArticulationAtomIntegrationTask*>(mFlushPool.allocate(sizeof(PxgArticulationAtomIntegrationTask)));

				task = PX_PLACEMENT_NEW(task, PxgArticulationAtomIntegrationTask)(
					articulationNodeIndices + a,
					PxMin(articulationBatchSize, mArticulationCount - a), &mCachedPositionIterations,
					&mCachedVelocityIterations, mIslandManager);

				task->setContinuation(continuation);
				task->removeReference();
			}
		}

		const PxU32 kinematicBatchSize = 1024u;
		const PxNodeIndex*const kinematicIndices = islandSim.getActiveKinematics();

		for (PxU32 a = 0; a < mKinematicCount; a += kinematicBatchSize)
		{
			PxgSetupKinematicTask* task = PX_PLACEMENT_NEW(mFlushPool.allocate(sizeof(PxgSetupKinematicTask)), PxgSetupKinematicTask)
				(kinematicIndices + a, mActiveNodeIndex.begin(), PxMin(mKinematicCount - a, kinematicBatchSize), mIslandManager, a + 1, mSolverBodyDataPool.begin() + a + 1,
					mSolverBodySleepDataPool.begin() + a + 1, mSolverTxIDataPool.begin() + a + 1);
			task->setContinuation(continuation);
			task->removeReference();
		}

		PxgSimulationController* gpuSimController = static_cast<PxgSimulationController*>(mSimulationController);
		//const PxU32 numParticles = gpuSimController->getNbParticleSystems();

		PxgBodySimManager& bodySimManager = gpuSimController->getBodySimManager();
		void** bodySimsLL = bodySimManager.mBodies.begin();
		
		PxI32 maxPosIters = 0, maxVelIters = 0;

		const PxU32 numParticleCores = mGpuParticleSystemCores.size();
		for (PxU32 i = 0; i < numParticleCores; ++i)
		{
			PxgParticleSystemCore* particleCore = mGpuParticleSystemCores[i];
			particleCore->getMaxIterationCount(bodySimManager, maxPosIters, maxVelIters);
		}

		{
			//Need to implement soft body
			PxU32* softBodyNodeIndex = gpuSimController->getSoftBodyNodeIndex();

			const PxU32 nbActiveSoftbodies = bodySimManager.mActiveSoftbodies.size();
			PxU32* activeSoftbodies = bodySimManager.mActiveSoftbodies.begin();

			for (PxU32 i = 0; i < nbActiveSoftbodies; ++i)
			{
				const PxU32 index = activeSoftbodies[i];
				const PxU32 nodeIdex = softBodyNodeIndex[index];
				Dy::DeformableVolume* dySoftBody = reinterpret_cast<Dy::DeformableVolume*>(bodySimsLL[nodeIdex]);

				const PxU16 solverIterationCounts = dySoftBody->getIterationCounts();

				maxPosIters = PxMax(PxI32(solverIterationCounts & 0xff), maxPosIters);
				maxVelIters = PxMax(PxI32(solverIterationCounts >> 8), maxVelIters);
			}
		}

		{
			// FEM cloth
			PxU32* femClothNodeIndex = gpuSimController->getFEMClothNodeIndex();

			const PxU32 nbActiveFEMCloths = bodySimManager.mActiveFEMCloths.size();
			PxU32* activeFEMCloths = bodySimManager.mActiveFEMCloths.begin();

			for (PxU32 i = 0; i < nbActiveFEMCloths; ++i)
			{
				const PxU32 index = activeFEMCloths[i];
				const PxU32 nodeIdex = femClothNodeIndex[index];
				Dy::DeformableSurface* dyFEMCloth = reinterpret_cast<Dy::DeformableSurface*>(bodySimsLL[nodeIdex]);

				const PxU16 solverIterationCounts = dyFEMCloth->getIterationCounts();

				maxPosIters = PxMax(PxI32(solverIterationCounts & 0xff), maxPosIters);
				//maxVelIters = PxMax(PxI32(solverIterationCounts >> 8), maxVelIters);
			}
		}

		PxAtomicMax(&mCachedPositionIterations, maxPosIters);
		PxAtomicMax(&mCachedVelocityIterations, maxVelIters);

		mGpuSolverCore->releaseContext();
	}

	void PxgGpuContext::doConstraintPrePrepCommon(physx::PxBaseTask* continuation)
	{
		mGpuSolverCore->acquireContext();

		m1dConstraintBatchIndices.forceSize_Unsafe(0);
		m1dConstraintBatchIndices.reserve(mIncrementalPartition.getNbConstraintBatches() + mNumStaticRigid1dConstraintBatches);

		mContactConstraintBatchIndices.forceSize_Unsafe(0);
		mContactConstraintBatchIndices.reserve(mIncrementalPartition.getNbContactBatches() + mNumStaticRigidContactBatches);

		mArti1dConstraintBatchIndices.forceSize_Unsafe(0);
		mArti1dConstraintBatchIndices.reserve(mIncrementalPartition.getNbArtiConstraintBatches() + mNumStaticArti1dConstraintBatches + mNumSelfArti1dConstraintBatches);

		mArtiContactConstraintBatchIndices.forceSize_Unsafe(0);
		mArtiContactConstraintBatchIndices.reserve(mIncrementalPartition.getNbArtiContactBatches() + mNumStaticArtiContactBatches + mNumSelfArtiContactBatches);

		mIslandContextPool[0].mNumPositionIterations = mCachedPositionIterations;
		mIslandContextPool[0].mNumVelocityIterations = mCachedVelocityIterations;

		mNum1dConstraintBatches = (PxI32)mIncrementalPartition.getNbConstraintBatches();
		mNumContactBatches = (PxI32)mIncrementalPartition.getNbContactBatches();
		mNumArtiContactBatches = (PxI32)mIncrementalPartition.getNbArtiContactBatches();
		mNumArti1dConstraintBatches = (PxI32)mIncrementalPartition.getNbArtiConstraintBatches();

		PxgBodySimManager& bodyManager = getSimulationController()->getBodySimManager();
		const PxU32 nbStaticSlabs = (PxMax(bodyManager.mMaxStaticRBJoints, bodyManager.mMaxStaticRBContacts) + mMaxNumStaticPartitions - 1) / mMaxNumStaticPartitions;

		const PxU32 maxCombinedSlabPartitions = mIncrementalPartition.getCombinedSlabMaxNbPartitions();

		mGpuSolverCore->gpuMemDmaUpBodyData(mSolverBodyDataPool, mSolverTxIDataPool, mIslandManager.getNbNodeHandles() + 1,
			mNumConstraintBatches, mNumArticConstraintBatches, PxMax(1u, (mIncrementalPartition.getNbPartitions() + maxCombinedSlabPartitions - 1) / maxCombinedSlabPartitions),
			nbStaticSlabs, mMaxNumStaticPartitions);

		//Allocate enough space for the friction patches now that we know how many we need after constraint partitioning
		{
			PX_PROFILE_ZONE("GpuDynamics.allocateFrictionPatchStreams", 0);
			mGpuSolverCore->allocateFrictionPatchStream(mNumContactBatches + mNumStaticRigidContactBatches, mNumArtiContactBatches + mNumStaticArtiContactBatches + mNumSelfArtiContactBatches);
		}

		mNum1DConstraintBlockPrepPool = (PxU32)mNum1dConstraintBatches;

		const PxU32 nbConstraintsPerBatch = mIsTGS ? PxgCpuConstraintPrePrepTask::NbConstraintsPerTaskTGS : PxgCpuConstraintPrePrepTask::NbConstraintsPerTaskPGS; //Each task processed up to PxgCpuConstraintPrePrepTask::NbConstraintsPerTask constraints of a certain type
		const PxU32 nbArtiConstraintsPerBatch = mIsTGS ? PxgCpuArtiConstraintPrePrepTask::NbConstraintsPerTaskTGS : PxgCpuArtiConstraintPrePrepTask::NbConstraintsPerTaskPGS;

		PxU32 constraintBatchIndex = 0;
		PxU32 contactBatchIndex = 0;
		PxU32 articulationConstraintBatchIndex = mNum1dConstraintBatches;
		PxU32 articulationContactBatchIndex = mNumContactBatches;

		const PxU32 batchMask = PXG_BATCH_SIZE - 1;

		mHasForceThresholds = mIncrementalPartition.hasForceThresholds();

		const PxInt32ArrayPinned& startSlabIter = mIncrementalPartition.getStartSlabPerPartition();
		const PxInt32ArrayPinned& articstartSlabIter = mIncrementalPartition.getArticStartSlabPerPartition();

		PxgJointManager& jointManager = static_cast<PxgSimulationController*>(mSimulationController)->getJointManager();
		const PxPinnedArray<PxgConstraintPrePrep>& rigidPreprepIter = jointManager.getGpuRigidJointPrePrep();
		const PxPinnedArray<PxgConstraintPrePrep>& artiPreprepIter = jointManager.getGpuArtiJointPrePrep();

		//The code below iterates over all partitions, producing tasks to fill in data.

		//Running indices

		PxU32 startIdx = 0; //Which partition to start at
		PxU32 startBatchOffset = 0;	//Batch offset within the partition
		PxU32 startOffset = 0; //Constraint offset within the partition
		PxU32 runningContactCount = 0; //The running count of the number of contact constraints that will be processed by the next task
		PxU32 runningBatchCount = 0; //The running count of the number of batches that will be processed by the next task

		{
			PX_PROFILE_ZONE("Process Partitions", 0);
			for (PxU32 i = 0; i < mIncrementalPartition.getNbPartitions(); ++i) // this is looping over "true" partitions, not the combined ones for the solver
			{
				const Partition& partition = mIncrementalPartition.getPartitionSlabs()[i / PXG_BATCH_SIZE]->mPartitions[i&(PXG_BATCH_SIZE - 1)];
				const PxU32 nbContacts = partition.mPartitionIndices[PxgEdgeType::eCONTACT_MANAGER].size();
				const PxU32 nbConstraints = partition.mPartitionIndices[PxgEdgeType::eCONSTRAINT].size();
				const PxU32 nbArtiContacts = partition.mPartitionIndices[PxgEdgeType::eARTICULATION_CONTACT].size();
				const PxU32 nbArtiConstraints = partition.mPartitionIndices[PxgEdgeType::eARTICULATION_CONSTRAINT].size();
				//PxU32* constraintIds = partition.mPartitionIndices[IG::Edge::eCONSTRAINT].begin();
				const PartitionIndices& constraintIds = partition.mPartitionIndices[PxgEdgeType::eCONSTRAINT];
				const PartitionIndices& artiConstraintIds = partition.mPartitionIndices[PxgEdgeType::eARTICULATION_CONSTRAINT];
				const PartitionIndices& artiContactIds = partition.mPartitionIndices[PxgEdgeType::eARTICULATION_CONTACT];
				const PxU32 jointStartIndex = mIncrementalPartition.getJointStartIndices()[i];

				PxU32 batchIndex = startSlabIter[i];
				PxU32 localArticBatchIndex = articstartSlabIter[i];
				PxU32 batchOffset = 0;

				for (PxU32 a = 0; a < nbConstraints; a += nbConstraintsPerBatch)
				{
					PxU32 nbConstraintsToProcess = PxMin(nbConstraints - a, nbConstraintsPerBatch);
					PxU32 nbBatches = (nbConstraintsToProcess + batchMask) / PXG_BATCH_SIZE;

					PxgCpuConstraintPrePrepTask* task = (PxgCpuConstraintPrePrepTask*)mFlushPool.allocate(sizeof(PxgCpuConstraintPrePrepTask));
					task = PX_PLACEMENT_NEW(task, PxgCpuConstraintPrePrepTask)(constraintIds, a, nbConstraintsToProcess,
						mConstraintBatchHeaders + batchIndex, nbBatches, constraintBatchIndex, jointStartIndex + a, mConstraintUniqueIndices,
						rigidPreprepIter.begin());

					task->setContinuation(continuation);
					task->removeReference();

					for (PxU32 b = 0; b < nbBatches; ++b)
					{
						PxU32 val = batchIndex + b;
						m1dConstraintBatchIndices.pushBack(val);
					}

					constraintBatchIndex += nbBatches;
					batchIndex += nbBatches;
				}

				PxU32 remainingContacts = nbContacts;

				PxU32 localOffset = 0;

				//While there are constraints in this partition, process them in chunks of ~nbConstraintsPerBatch
				while ((runningContactCount + remainingContacts) >= nbConstraintsPerBatch)
				{
					//We are aiming to process approximately 2048 constraints. However, to simplify the logic in the CPU PrePrep task, 
					//we actually can process a little more than that to fill up entire batches. Each batch contains 32 constraints.
					PxU32 nbConstraintsFromThisPartition = nbConstraintsPerBatch - runningContactCount; //Number of constraints from this partition
					PxU32 nbBatchesFromThisPartition = ((nbConstraintsFromThisPartition + batchMask) / PXG_BATCH_SIZE); //The number of batches from this partition (groups of 32 constraints)

					//Round up the number of constraints from this partition to be full batches unless there are insufficient constraints in this partition to create a full batch
					nbConstraintsFromThisPartition = PxMin((nbConstraintsFromThisPartition + batchMask)&(~(batchMask)), remainingContacts);

					PxU32 totalBatches = runningBatchCount + nbBatchesFromThisPartition;

					PxU32 nbConstraintsToProcess = runningContactCount + nbConstraintsFromThisPartition;

					PxgCpuContactPrePrepTask* task = (PxgCpuContactPrePrepTask*)mFlushPool.allocate(sizeof(PxgCpuContactPrePrepTask));
					task = PX_PLACEMENT_NEW(task, PxgCpuContactPrePrepTask)(mIncrementalPartition, startIdx, startOffset, nbConstraintsToProcess,
						startSlabIter.begin(), startBatchOffset, mIncrementalPartition.getContactStartIndices().begin(),
						mConstraintBatchHeaders, totalBatches, contactBatchIndex, mContactUniqueIndices,
						mOutputIterator, mPatchStreamAllocators[mCurrentContactStream]->mStart,
						mContactStreamAllocators[mCurrentContactStream]->mStart);

					task->setContinuation(continuation);
					task->removeReference();

					//Update contact counts
					remainingContacts -= nbConstraintsFromThisPartition;
					localOffset += nbConstraintsFromThisPartition;

					for (PxU32 b = 0; b < nbBatchesFromThisPartition; ++b)
					{
						PxU32 val = batchIndex + b;
						mContactConstraintBatchIndices.pushBack(val);
					}
					//Update iteration indices in this partition
					contactBatchIndex += totalBatches;
					batchIndex += nbBatchesFromThisPartition;
					batchOffset += nbBatchesFromThisPartition;

					//Update global task iteration indices
					startIdx = i;
					startOffset = localOffset;
					startBatchOffset = batchOffset;
					runningContactCount = 0;
					runningBatchCount = 0;
				}

				//We have remaining constraints. If so, sum them up and continue iterating...
				PxU32 remainingBatches = (remainingContacts + batchMask) / PXG_BATCH_SIZE;
				runningContactCount += remainingContacts;
				runningBatchCount += remainingBatches;

				for (PxU32 b = 0; b < remainingBatches; ++b)
				{
					PxU32 val = batchIndex + b;
					mContactConstraintBatchIndices.pushBack(val);
				}

				//batchIndex += runningBatchCount;

				PxU32 localArtiJointStartIndex = mIncrementalPartition.getArtiJointStartIndices()[i];

				//constraintBatchIndex += contactBatchIndex;

				//articulation constraints
				for (PxU32 a = 0; a < nbArtiConstraints; a += nbArtiConstraintsPerBatch)
				{
					//each constraint is a batch
					PxU32 nbConstraintsToProcess = PxMin(nbArtiConstraints - a, nbArtiConstraintsPerBatch);
					PxU32 nbBatchesFromThisPartition = ((nbConstraintsToProcess + batchMask) / PXG_BATCH_SIZE); //The number of batches from this partition (groups of 32 constraints)

					PxgCpuArtiConstraintPrePrepTask* task = (PxgCpuArtiConstraintPrePrepTask*)mFlushPool.allocate(sizeof(PxgCpuArtiConstraintPrePrepTask));
					task = PX_PLACEMENT_NEW(task, PxgCpuArtiConstraintPrePrepTask)(artiConstraintIds, a, nbConstraintsToProcess,
						mArticConstraintBatchHeaders + localArticBatchIndex, nbBatchesFromThisPartition, articulationConstraintBatchIndex, localArtiJointStartIndex, mArtiConstraintUniqueIndices,
						artiPreprepIter.begin(), false);

					localArtiJointStartIndex += nbConstraintsToProcess;

					task->setContinuation(continuation);
					task->removeReference();

					for (PxU32 b = 0; b < nbBatchesFromThisPartition; ++b)
					{
						PxU32 val = localArticBatchIndex + b;
						mArti1dConstraintBatchIndices.pushBack(val);
					}

					articulationConstraintBatchIndex += nbBatchesFromThisPartition;
					localArticBatchIndex += nbBatchesFromThisPartition;
				}

				PxU32 localArtiContactStartIndex = mIncrementalPartition.getArtiContactStartIndices()[i];
				//articulation contacts
				for (PxU32 a = 0; a < nbArtiContacts; a += nbArtiConstraintsPerBatch)
				{
					//each contact is a batch
					PxU32 nbContactsToProcess = PxMin(nbArtiContacts - a, nbArtiConstraintsPerBatch);
					PxU32 nbBatchesFromThisPartition = ((nbContactsToProcess + batchMask) / PXG_BATCH_SIZE); //The number of batches from this partition (groups of 32 constraints)

					PxgCpuArtiConstraintPrePrepTask* task = (PxgCpuArtiConstraintPrePrepTask*)mFlushPool.allocate(sizeof(PxgCpuArtiConstraintPrePrepTask));
					task = PX_PLACEMENT_NEW(task, PxgCpuArtiConstraintPrePrepTask)(artiContactIds, a, nbContactsToProcess,
						mArticConstraintBatchHeaders + localArticBatchIndex, nbBatchesFromThisPartition, articulationContactBatchIndex, localArtiContactStartIndex, mArtiContactUniqueIndices,
						artiPreprepIter.begin(), true);

					localArtiContactStartIndex += nbContactsToProcess;

					task->setContinuation(continuation);
					task->removeReference();

					for (PxU32 b = 0; b < nbBatchesFromThisPartition; ++b)
					{
						PxU32 val = localArticBatchIndex + b;
						mArtiContactConstraintBatchIndices.pushBack(val);
					}

					articulationContactBatchIndex += nbBatchesFromThisPartition;
					localArticBatchIndex += nbBatchesFromThisPartition;
				}
			}

			if (runningBatchCount > 0)
			{
				//There are remaining unprocessed contact constraints
				PxgCpuContactPrePrepTask* task = (PxgCpuContactPrePrepTask*)mFlushPool.allocate(sizeof(PxgCpuContactPrePrepTask));
				task = PX_PLACEMENT_NEW(task, PxgCpuContactPrePrepTask)(mIncrementalPartition, startIdx, startOffset, runningContactCount,
					startSlabIter.begin(), startBatchOffset, mIncrementalPartition.getContactStartIndices().begin(),
					mConstraintBatchHeaders, runningBatchCount, contactBatchIndex, mContactUniqueIndices,
					mOutputIterator, mPatchStreamAllocators[mCurrentContactStream]->mStart,
					mContactStreamAllocators[mCurrentContactStream]->mStart);

				task->setContinuation(continuation);
				task->removeReference();
			}
		}

		doStaticArticulationConstraintPrePrep(continuation, articulationConstraintBatchIndex, articulationContactBatchIndex);
		doStaticRigidConstraintPrePrep(continuation);

		mGpuSolverCore->releaseContext();
	}

	void PxgGpuContext::doConstraintPrePrepGPUCommon(bool hasForceThresholds)
	{
		mLostTouchTask->removeReference();

		const PxU32	nbCombinedSlabPartitions = mIncrementalPartition.getCombinedSlabNbPartitions();

		{
			mConstraintsPerPartition.forceSize_Unsafe(0);
			if (mConstraintsPerPartition.capacity() < nbCombinedSlabPartitions)
				mConstraintsPerPartition.reserve(2 * nbCombinedSlabPartitions);

			mArtiConstraintsPerPartition.forceSize_Unsafe(0);
			if (mArtiConstraintsPerPartition.capacity() < nbCombinedSlabPartitions)
				mArtiConstraintsPerPartition.reserve(2 * nbCombinedSlabPartitions);

			for (PxU32 a = 0; a < nbCombinedSlabPartitions; ++a)
			{
				mConstraintsPerPartition.pushBack(mIncrementalPartition.getCSlabAccumulatedPartitionCount(a));
				mArtiConstraintsPerPartition.pushBack(mIncrementalPartition.getCSlabAccumulatedArtiPartitionCount(a));
			}
		}

		mIslandContextPool->mStartPartitionIndex = 0;
		mIslandContextPool->mNumPartitions = nbCombinedSlabPartitions;
		mIslandContextPool->mBatchStartIndex = 0;
		mIslandContextPool->mBatchCount = mIncrementalPartition.getNbConstraintBatches() + mIncrementalPartition.getNbContactBatches();

		mIslandContextPool->mArtiBatchStartIndex = 0;
		mIslandContextPool->mArtiBatchCount = mIncrementalPartition.getNbArtiConstraintBatches() + mIncrementalPartition.getNbArtiContactBatches();
		//mIslandContextPool->mStaticArtiBatchCount = getSimulationController()->getBodySimManager().mTotalArticJoints + getSimulationController()->getBodySimManager().mTotalArticContacts;

		PxgJointManager& jointManager = getSimulationController()->getJointManager();
		const PxU32 gpuRigidJointSize = jointManager.getGpuNbRigidConstraints();
		const PxU32 cpuRigidJointSize = jointManager.getCpuNbRigidConstraints();
		const PxU32 gpuArtiJointSize = jointManager.getGpuNbArtiConstraints();
		const PxU32 cpuArtiJointSize = jointManager.getCpuNbArtiConstraints();
		PxgConstraintPrePrepData ppData;
		ppData.nbGpuRigidJoints = gpuRigidJointSize;
		ppData.nbTotalRigidJoints = gpuRigidJointSize + cpuRigidJointSize;
		ppData.nbGpuArtiJoints = gpuArtiJointSize;
		ppData.nbTotalArtiJoints = gpuArtiJointSize + cpuArtiJointSize;

		ppData.numContactBatches = PxU32(mNumContactBatches);
		ppData.num1dConstraintBatches = PxU32(mNum1dConstraintBatches);
		ppData.numStaticContactBatches = PxU32(mNumStaticRigidContactBatches);
		ppData.numStatic1dConstraintBatches = PxU32(mNumStaticRigid1dConstraintBatches);

		ppData.numArtiContactsBatches = PxU32(mNumArtiContactBatches);
		ppData.numArti1dConstraintBatches = PxU32(mNumArti1dConstraintBatches);
		ppData.numArtiStaticContactsBatches = PxU32(mNumStaticArtiContactBatches);
		ppData.numArtiStatic1dConstraintBatches = PxU32(mNumStaticArti1dConstraintBatches);
		ppData.numArtiSelfContactsBatches = PxU32(mNumSelfArtiContactBatches);
		ppData.numArtiSelf1dConstraintBatches = PxU32(mNumSelfArti1dConstraintBatches);

		ppData.artiStaticConstraintBatchOffset = PxU32(mArtiStaticConstraintBatchOffset);
		ppData.artiStaticContactBatchOffset = PxU32(mArtiStaticContactBatchOffset);

		ppData.contactUniqueIndices = mContactUniqueIndices;
		ppData.constraintUniqueIndices = mConstraintUniqueIndices;
		ppData.artiContactUniqueIndices = mArtiContactUniqueIndices;
		ppData.artiConstraintUniqueindices = mArtiConstraintUniqueIndices;
		if (mIncrementalPartition.getStaticContactsOnly())
		{
			// Phase B: GPU kernel device buffer pointers
			ppData.artiStaticContactUniqueIndices = reinterpret_cast<PxU32*>(mStaticContactIndices_d);
			ppData.artiStaticContactCount = reinterpret_cast<PxU32*>(mStaticContactCounts_d);
			ppData.artiStaticConstraintUniqueIndices = NULL;
			ppData.artiStaticConstraintCount = NULL;
			ppData.artiStaticConstraintStartIndex = NULL;
			ppData.artiStaticContactStartIndex = NULL;
		}
		else
		{
			ppData.artiStaticConstraintUniqueIndices = mArtiStaticConstraintUniqueIndices;
			ppData.artiStaticContactUniqueIndices = mArtiStaticContactUniqueIndices;
			ppData.artiStaticConstraintStartIndex = mArtiStaticConstraintStartIndex;
			ppData.artiStaticConstraintCount = mArtiStaticConstraintCount;
			ppData.artiStaticContactStartIndex = mArtiStaticContactStartIndex;
			ppData.artiStaticContactCount = mArtiStaticContactCount;
		}

		ppData.constraint1DBatchIndices = m1dConstraintBatchIndices.begin();
		ppData.constraintContactBatchIndices = mContactConstraintBatchIndices.begin();
		ppData.artiConstraint1dBatchindices = mArti1dConstraintBatchIndices.begin();
		ppData.artiConstraintContactBatchIndices = mArtiContactConstraintBatchIndices.begin();

		PxgConstantData cData;
		cData.dt = mDt;
		cData.invDtF32 = mInvDt;
		cData.bounceThresholdF32 = mBounceThreshold;
		cData.frictionOffsetThreshold = mFrictionOffsetThreshold;
		cData.correlationDistance = mCorrelationDistance;
		cData.ccdMaxSeparation = mCCDSeparationThreshold;
		cData.biasCoefficient = mIslandContextPool->mBiasCoefficient;
		cData.gravity = mGravity;

		PxgBodySimManager& bodySimManager = getSimulationController()->getBodySimManager();

		PxgPartitionData pData;
		pData.constraintsPerPartition = mConstraintsPerPartition.begin();
		pData.numConstraintsPerPartition = mConstraintsPerPartition.size();
		pData.artiConstraintsPerPartition = mArtiConstraintsPerPartition.begin();
		pData.numArtiConstraintsPerPartition = mArtiConstraintsPerPartition.size();
		pData.numTotalContacts = mIncrementalPartition.getTotalContacts();
		pData.numTotalStaticConstraints = bodySimManager.mTotalStaticRBJoints;
		pData.numTotalStaticContacts = bodySimManager.mTotalStaticRBContacts;
		pData.numTotalConstraints = mIncrementalPartition.getTotalConstraints();
		pData.numTotalArtiContacts = mIncrementalPartition.getTotalArticulationContacts();
		pData.numTotalArtiConstraints = mIncrementalPartition.getTotalArticulationConstraints();
		pData.numTotalArtiStaticContacts = bodySimManager.mTotalStaticArticContacts;
		pData.numTotalArtiStaticConstraints = bodySimManager.mTotalStaticArticJoints;
		pData.numTotalArtiSelfContacts = bodySimManager.mTotalSelfArticContacts;
		pData.numTotalArtiSelfConstraints = bodySimManager.mTotalSelfArticJoints;
		pData.artiStaticConstraintBatchOffset = mArtiStaticConstraintBatchOffset;
		pData.artiStaticContactBatchOffset = mArtiStaticContactBatchOffset;

		mIslandContextPool->mStaticArtiBatchCount = mNumArtiStaticConstraintBatches;
		mIslandContextPool->mSelfArtiBatchCount = mNumArtiSelfConstraintBatches;
		mIslandContextPool->mStaticRigidBatchCount = mNumRigidStaticConstraintBatches;

		const PxU32 maxCombinedSlabPartitions = mIncrementalPartition.getCombinedSlabMaxNbPartitions();
		const PxU32 nbSlabs = PxMax(1u, (mIncrementalPartition.getNbPartitions() + maxCombinedSlabPartitions - 1) / maxCombinedSlabPartitions);
		const PxU32 nbPartitions = PxMin(mIncrementalPartition.getNbPartitions(), maxCombinedSlabPartitions);

		mGpuArticulationCore->allocDeltaVBuffer(nbSlabs, nbPartitions, mGpuSolverCore->getStream());

		{
			mGpuSolverCore->gpuMemDMAUp(*mPinnedMemoryAllocator, ppData, mSolverBodyPool.size(),
				mConstraintBatchHeaders, mIslandContextPool, mNumIslandContextPool, pData,
				mNumConstraintBatches, mNumRigidStaticConstraintBatches, mNumArticConstraintBatches, mNumArtiStaticConstraintBatches, mNumArtiSelfConstraintBatches, cData,
				PXG_MAX_NUM_POINTS_PER_CONTACT_PATCH * (mNumContactBatches + mNumStaticRigidContactBatches), 4u * (mNumContactBatches + mNumStaticRigidContactBatches),
				PXG_MAX_NUM_POINTS_PER_CONTACT_PATCH * (mNumArtiContactBatches + mNumStaticArtiContactBatches + mNumSelfArtiContactBatches), 4u * (mNumArtiContactBatches + mNumStaticArtiContactBatches + mNumSelfArtiContactBatches),
				mTotalEdges, mTotalPreviousEdges,
				nbSlabs,
				maxCombinedSlabPartitions, mEnableStabilization, mPatchStreamAllocators[mCurrentContactStream]->mStart, mContactStreamAllocators[mCurrentContactStream]->mStart,
				mForceStreamAllocator->mStart, mOutputIterator, mSolverBodyPool.size() - (mKinematicCount + 1), mKinematicCount + 1, mArticulationCount,
				reinterpret_cast<Cm::UnAlignedSpatialVector*>(mGpuArticulationCore->getDeferredZ()),
				reinterpret_cast<PxU32*>(mGpuArticulationCore->getArticulationDirty()),
				reinterpret_cast<uint4*>(mGpuArticulationCore->getArticulationSlabMask()),
				mGPUShapeInteractions, mGPURestDistances, mGPUTorsionalData, mArtiStaticContactIndices.begin(), mArtiStaticContactIndices.size(),
				mArtiStaticJointIndices.begin(), mArtiStaticJointIndices.size(), mArtiStaticContactCounts.begin(), mArtiStaticJointCounts.begin(),
				mArtiSelfContactIndices.begin(), mArtiSelfContactIndices.size(),
				mArtiSelfJointIndices.begin(), mArtiSelfJointIndices.size(), mArtiSelfContactCounts.begin(), mArtiSelfJointCounts.begin(),
				mRigidStaticContactIndices.begin(), mRigidStaticContactIndices.size(), mRigidStaticJointIndices.begin(), mRigidStaticJointIndices.size(),
				mRigidStaticContactCounts.begin(), mRigidStaticJointCounts.begin(), mLengthScale, hasForceThresholds);
		}

		//Make sure that the GPU articulation work has completed now...
		mGpuArticulationCore->syncUnconstrainedVelocities();
		mGpuArticulationCore->layoutDeltaVBuffer(nbSlabs, nbPartitions, mGpuSolverCore->getStream());

		mGpuArticulationCore->createStaticContactAndConstraintsBatch(mArticulationCount);

		mGpuSolverCore->constraintPrePrepParallel(mNumConstraintBatches + mNumRigidStaticConstraintBatches + mNumArticConstraintBatches + mNumArtiStaticConstraintBatches + mNumArtiSelfConstraintBatches, gpuRigidJointSize + gpuArtiJointSize,
			mIslandContextPool->mBodyCount);
	}

	void PxgCpuJointPrePrepTask::runInternal()
	{
		PxU32 endIndex = mStartIndex + mNbToProcess;

		Px1DConstraint tempRows[Dy::MAX_CONSTRAINT_ROWS];

		for (PxU32 i = mStartIndex; i < endIndex; ++i)
		{
			const Dy::Constraint* constraint = mConstraints[i];

			const PxConstraintSolverPrep solverPrep = constraint->solverPrep;

			if (!solverPrep)
				continue;

			const PxTransform& pose0 = (constraint->body0 ? constraint->body0->getPose() : PxTransform(PxIdentity));
			const PxTransform& pose1 = (constraint->body1 ? constraint->body1->getPose() : PxTransform(PxIdentity));
			const void* constantBlock = constraint->constantBlock;

			PxgConstraintData& data = mConstraintData[i];
			//Px1DConstraint* rows = &rowIter[i*Dy::MAX_CONSTRAINT_ROWS];

			PxMemZero(tempRows, sizeof(Px1DConstraint)*Dy::MAX_CONSTRAINT_ROWS);

			for (PxU32 j = 0; j < Dy::MAX_CONSTRAINT_ROWS; j++)
			{
				Px1DConstraint& c = tempRows[j];
				c.minImpulse = -PX_MAX_REAL;
				c.maxImpulse = PX_MAX_REAL;
			}

			PxConstraintInvMassScale ims(1.0f, 1.0f, 1.0f, 1.0f);
			PxVec3p ra, rb;
			PxVec3p body0WorldOffset(0.0f);

			//TAG:solverprepcall
			const PxU32 numRows = (constraint->flags & PxConstraintFlag::eDISABLE_CONSTRAINT) ? 0 :(*solverPrep)(tempRows,
				body0WorldOffset,
				Dy::MAX_CONSTRAINT_ROWS,
				ims,
				constantBlock,
				pose0, pose1, !!(constraint->flags & PxConstraintFlag::eENABLE_EXTENDED_LIMITS), ra, rb);

			data.mNumRows_Flags_StartIndex.x = numRows;

			if (numRows == 0)
				continue;

			ra -= pose0.p;
			rb -= pose1.p;

			data.mInvMassScale.linear0 = ims.linear0;
			data.mInvMassScale.angular0 = ims.angular0;
			data.mInvMassScale.linear1 = ims.linear1;
			data.mInvMassScale.angular1 = ims.angular1;
			data.mRaWorld_linBreakForceW = make_float4(ra.x, ra.y, ra.z, constraint->linBreakForce);
			data.mRbWorld_angBreakForceW = make_float4(rb.x, rb.y, rb.z, constraint->angBreakForce);

			data.mNumRows_Flags_StartIndex.y = constraint->flags;

			PxI32 startRowIndex = PxAtomicAdd(mRowCounts, PxI32(numRows)) - PxI32(numRows);

			PxMemCopy(mConstraintRows + startRowIndex, tempRows, sizeof(Px1DConstraint) * numRows);

			data.mNumRows_Flags_StartIndex.z = mGpuJointOffset + startRowIndex;
		}
	}

	void PxgGpuContext::cpuJointPrePrepTask(physx::PxBaseTask* continuation)
	{
		PxgJointManager& jointManager = getSimulationController()->getJointManager();

		// AD: This could also be skipped with direct-GPU API, but at this point the constraints are already partitioned and I
		// cannot figure out how to remove the CPU joints from there again.

		const PxArray<const Dy::Constraint*>& cpuRigidConstraints = jointManager.getCpuRigidConstraints();
		const PxArray<const Dy::Constraint*>& cpuArtiConstraints = jointManager.getCpuArtiConstraints();

		const PxU32 nbCpuRigidConstraints = cpuRigidConstraints.size();
		const PxU32 nbCpuArtiConstraints = cpuArtiConstraints.size();

		const PxU32 gpuRigidJointOutputOffset = jointManager.getGpuNbRigidConstraints() * Dy::MAX_CONSTRAINT_ROWS;

		const PxU32 nbJointsPerTask = 128u;	// PT: TODO: revisit
		//for other joint
		for (PxU32 a = 0; a < nbCpuRigidConstraints; a += nbJointsPerTask)
		{
			const PxU32 nbToProcess = PxMin(nbCpuRigidConstraints - a, nbJointsPerTask);
			PxgCpuJointPrePrepTask* task = reinterpret_cast<PxgCpuJointPrePrepTask*>(mFlushPool.allocate(sizeof(PxgCpuJointPrePrepTask)));
			task = PX_PLACEMENT_NEW(task, PxgCpuJointPrePrepTask)(*getSimulationController(), a, nbToProcess, gpuRigidJointOutputOffset,
				cpuRigidConstraints.begin(), jointManager.getCpuRigidConstraintData().begin(), jointManager.getCpuRigidConstraintRows().begin(),
				&jointManager.mNbCpuRigidConstraintRows);

			task->setContinuation(continuation);
			task->removeReference();
		}

		const PxU32 gpuArtiJointOutputOffset = jointManager.getGpuNbArtiConstraints() * Dy::MAX_CONSTRAINT_ROWS;

		for (PxU32 a = 0; a < nbCpuArtiConstraints; a += nbJointsPerTask)
		{
			const PxU32 nbToProcess = PxMin(nbCpuArtiConstraints - a, nbJointsPerTask);
			PxgCpuJointPrePrepTask* task = reinterpret_cast<PxgCpuJointPrePrepTask*>(mFlushPool.allocate(sizeof(PxgCpuJointPrePrepTask)));
			task = PX_PLACEMENT_NEW(task, PxgCpuJointPrePrepTask)(*getSimulationController(), a, nbToProcess, gpuArtiJointOutputOffset,
				cpuArtiConstraints.begin(), jointManager.getCpuArtiConstraintData().begin(),
				jointManager.getCpuArtiConstraintRows().begin(), &jointManager.mNbCpuArtiConstraintRows);

			task->setContinuation(continuation);
			task->removeReference();
		}
	}

	// This class figures out the max iteration counts for all actors,
	// and prepares some data for kinematics.
	void PxgCpuPreIntegrationTask::runInternal()
	{
		nvtxRangePush("px:cpuPreIntegration");
		mContext.doPreIntegrationTaskCommon(mCont);
		nvtxRangePop();
	}

	void PxgCpuContactPrePrepTask::runInternal()
	{
		nvtxRangePush("px:cpuContactPrePrep");
		PX_PROFILE_ZONE("GpuDynamics.PxgCpuContactPrePrepTask", 0);

		const PxU32 nbToProcess = mNumBatches;
		PxU32 nbProcessed = 0;
		PxU32 partitionIdx = mPartitionIndex;
		PxU32 partitionStartIdx = mStartIndexWithinPartition;
		PxU32 startSlabOffset = mStartSlabOffset;

		PxU32 workUnitIndex = mWorkUnitStartIndex;

		while (nbProcessed < nbToProcess)
		{
			//Extract current partition
			const Partition& partition = mPartition.getPartitionSlabs()[partitionIdx / PXG_BATCH_SIZE]->mPartitions[partitionIdx&(PXG_BATCH_SIZE - 1)];
			//Get edgeIndices corresponding to this partition offset by partitionStartIdx
			const PartitionIndices& edgeIds = partition.mPartitionIndices[PxgEdgeType::eCONTACT_MANAGER];// +partitionStartIdx;

			//Factor in joint constraints to work out offsets in this partition. As this task can now process multiple partitions,
			//it is easiest just to compute them again here
			{
				const PxU32 nbConstraints = partition.mPartitionIndices[PxgEdgeType::eCONSTRAINT].size();
				const PxU32 nbBatches = (nbConstraints + 31u) / PXG_BATCH_SIZE;
				startSlabOffset += nbBatches;
			}

			const PxU32 batchIndex = mStartSlabIter[partitionIdx] + startSlabOffset;
			const PxU32 uniqueStartIndex = mContactStartIndices[partitionIdx] + partitionStartIdx;

			//The number we process in this partition is equal to the smaller of (nbToProcess - nbProcessed) and (size of partition - startOffsetInPartition).
			const PxU32 nbRemaining = partition.mPartitionIndices[PxgEdgeType::eCONTACT_MANAGER].size() - partitionStartIdx;
			//Convert from constraints to batches
			const PxU32 nbBatchesToProcess = PxMin((nbToProcess - nbProcessed), (nbRemaining + 31) / PXG_BATCH_SIZE);

			PxU32 currentEdgeIndex = 0;

			for (PxU32 a = 0; a < nbBatchesToProcess; ++a)
			{
				const PxU32 descStride = PxMin(nbRemaining - currentEdgeIndex, PXG_BATCH_SIZE);

				PxgConstraintBatchHeader& batchHeader = mBatchHeaders[a + batchIndex];
				batchHeader.constraintType = PxgSolverConstraintDesc::eCONTACT;
				batchHeader.mDescStride = PxU16(descStride);
				batchHeader.mConstraintBatchIndex = workUnitIndex++;
				batchHeader.mStartPartitionIndex = uniqueStartIndex + a * PXG_BATCH_SIZE;
				batchHeader.mask = 0xFFFFFFFF; //Unused

#if	PXG_CONTACT_VALIDATION
				validateContactPairs(a, a + descStride, edgeIds + a, mNpIds, mOutputIterator, mBaseContactPatch, mBaseContactPointer);
#endif
				currentEdgeIndex += descStride;
			}

			for (PxU32 i = 0; i < nbRemaining; ++i)
			{
				const PxU32 uniqueId = edgeIds[i + partitionStartIdx];
				mPinnedEdgeIds[uniqueStartIndex + i] = uniqueId;
			}

			nbProcessed += nbBatchesToProcess;
			partitionIdx++;
			partitionStartIdx = 0;
			startSlabOffset = 0;

			//PxMemCopy(mPinnedEdgeIds + uniqueStartIndex, edgeIds, sizeof(PxU32) * nbRemaining);
		}
		nvtxRangePop();
	}

	void PxgGpuContext::allocateTempPinnedSolverMemoryCommon()
	{
		// AD: two stages.
		// 1. first figure out how much we need. Allocate PxMax(sizeNeeded, PxGpuDynamicsMemoryConfig::tempBufferCapacity).
		// 2. suballocate and set the pointers.

		// AD: old comment that moved here when outlining into a separate function. I don't know how relevant this still is.
		// KS - this may be over-allocating because, at this stage, we only know (1) how many articulation static contacts
		// we have in total, (2) how many is the max a given articulation has and (3) how many articulations we have.
		// We allocate the minimum of maxBatches * numArticulations, totalContacts. We will likely require less than
		// both of these counts, but this provides us with an upper-bound...

		// this code operates under the assumption that we only have 1 solver island on GPU.

		PxU64 sizeNeeded = 0;
		const PxU32 alignment = 128; // GPU cache line size.

		const PxU32 totalIslands = 1;
		const PxU64 totalIslandsAllocationSize = (totalIslands * sizeof(PxgIslandContext)) + alignment;
		sizeNeeded += totalIslandsAllocationSize;

		mNumConstraintBatches = mIncrementalPartition.getNbConstraintBatches() + mIncrementalPartition.getNbContactBatches();

		PxgBodySimManager& bodyManager = getSimulationController()->getBodySimManager();
		const PxU32 maxStaticRigidJoints = bodyManager.mMaxStaticRBJoints;
		const PxU32 maxStaticRigidContacts = bodyManager.mMaxStaticRBContacts;
		const PxU32 nbRigidBatches = (mBodyCount + PXG_BATCH_SIZE - 1) / PXG_BATCH_SIZE;
		const PxU32 totalStaticRigidContacts = bodyManager.mTotalStaticRBContacts;
		const PxU32 totalStaticRigidJoints = bodyManager.mTotalStaticRBJoints;

		mNumStaticRigidContactBatches = PxMin(maxStaticRigidContacts * nbRigidBatches, totalStaticRigidContacts);
		mNumStaticRigid1dConstraintBatches = PxMin(maxStaticRigidJoints * nbRigidBatches, totalStaticRigidJoints);
		mNumRigidStaticConstraintBatches = (mNumStaticRigidContactBatches + mNumStaticRigid1dConstraintBatches);

		mNumArticConstraintBatches = mIncrementalPartition.getNbArtiConstraintBatches() + mIncrementalPartition.getNbArtiContactBatches();

		const PxU32 nbArticBatches = (mArticulationCount + PXG_BATCH_SIZE - 1) / PXG_BATCH_SIZE;
		const PxU32 maxStaticArticJoints = bodyManager.mMaxStaticArticJoints;
		const PxU32 maxStaticArticContacts = bodyManager.mMaxStaticArticContacts;
		const PxU32 totalStaticArticulationContacts = bodyManager.mTotalStaticArticContacts;
		const PxU32 totalStaticArticulationJoints = bodyManager.mTotalStaticArticJoints;

		mNumStaticArtiContactBatches = PxMin(maxStaticArticContacts * nbArticBatches, totalStaticArticulationContacts);
		mNumStaticArti1dConstraintBatches = PxMin(maxStaticArticJoints * nbArticBatches, totalStaticArticulationJoints);
		mNumArtiStaticConstraintBatches = (mNumStaticArtiContactBatches + mNumStaticArti1dConstraintBatches);

		const PxU32 maxSelfArticJoints = bodyManager.mMaxSelfArticJoints;
		const PxU32 maxSelfArticContacts = bodyManager.mMaxSelfArticContacts;
		const PxU32 totalSelfArticulationContacts = bodyManager.mTotalSelfArticContacts;
		const PxU32 totalSelfArticulationJoints = bodyManager.mTotalSelfArticJoints;

		mNumSelfArtiContactBatches = PxMin(maxSelfArticContacts * nbArticBatches, totalSelfArticulationContacts);
		mNumSelfArti1dConstraintBatches = PxMin(maxSelfArticJoints * nbArticBatches, totalSelfArticulationJoints);
		mNumArtiSelfConstraintBatches = (mNumSelfArtiContactBatches + mNumSelfArti1dConstraintBatches);

		const PxU64 allocationSizeConstraintBatchHeader = sizeof(PxgConstraintBatchHeader) * (mNumConstraintBatches + mNumRigidStaticConstraintBatches + mNumArticConstraintBatches + mNumArtiStaticConstraintBatches + mNumArtiSelfConstraintBatches);
		const PxU64 allocationSizeConstraintBatchHeaderAligned = allocationSizeConstraintBatchHeader + alignment;
		sizeNeeded += allocationSizeConstraintBatchHeaderAligned;
		
		const PxU32 totalJoints = mIncrementalPartition.getTotalConstraints();
		const PxU32 totalContacts = mIncrementalPartition.getTotalContacts();
		const PxU32 totalArticulationJoints = mIncrementalPartition.getTotalArticulationConstraints();
		const PxU32 totalArticulationContacts = mIncrementalPartition.getTotalArticulationContacts();

		//Unique Indices layout is joint->contact->artiJoint->artiContact
		const PxU64 allocationSizeUniqueIndices = (totalJoints + totalContacts + totalArticulationJoints
			+ totalArticulationContacts + totalStaticArticulationJoints + totalStaticArticulationContacts + totalSelfArticulationContacts
			+ totalSelfArticulationJoints + totalStaticRigidContacts + totalStaticRigidJoints) * sizeof(PxU32);
		const PxU64 allocationSizeUniqueIndicesAligned = allocationSizeUniqueIndices + alignment;
		sizeNeeded += allocationSizeUniqueIndicesAligned;

		const PxU64 allocationSizeArticulationCount = mArticulationCount * 4 * sizeof(PxU32);
		const PxU64 allocationSizeArticulationCountAligned = allocationSizeArticulationCount + alignment;
		sizeNeeded += allocationSizeArticulationCountAligned;

		const PxU64 allocationSizeBodyCount = mBodyCount * 2 * sizeof(PxU32);
		const PxU64 allocationSizeBodyCountAligned = allocationSizeBodyCount + alignment;
		sizeNeeded += allocationSizeBodyCountAligned;

		// descriptors are part of the solvercore
		sizeNeeded += mGpuSolverCore->getDescriptorsAllocationSize();

		// phase 2 - actually allocate the memory
		mPinnedMemoryAllocator->reserveAndGrow(static_cast<PxU32>(sizeNeeded));

#if PX_ENABLE_SIM_STATS
		mSimStats.mGpuDynamicsTempBufferCapacity = PxMax(sizeNeeded,mSimStats.mGpuDynamicsTempBufferCapacity);
#else
		PX_CATCH_UNDEFINED_ENABLE_SIM_STATS
#endif

		mIslandContextPool = reinterpret_cast<PxgIslandContext*>(mPinnedMemoryAllocator->allocate(totalIslands * sizeof(PxgIslandContext), alignment));

		mConstraintBatchHeaders = reinterpret_cast<PxgConstraintBatchHeader*>(mPinnedMemoryAllocator->allocate(allocationSizeConstraintBatchHeader, alignment));
		mArticConstraintBatchHeaders = mConstraintBatchHeaders + mNumConstraintBatches;

		mConstraintUniqueIndices = reinterpret_cast<PxU32*>(mPinnedMemoryAllocator->allocate(allocationSizeUniqueIndices, alignment));
		mRigidStaticConstraintUniqueIndices = mConstraintUniqueIndices + totalJoints;
		mArtiConstraintUniqueIndices = mRigidStaticConstraintUniqueIndices + totalStaticRigidJoints;
		mArtiStaticConstraintUniqueIndices = mArtiConstraintUniqueIndices + totalArticulationJoints;
		mArtiSelfConstraintUniqueIndices = mArtiStaticConstraintUniqueIndices + totalStaticArticulationJoints;
		
		mContactUniqueIndices = mArtiSelfConstraintUniqueIndices + totalSelfArticulationJoints;
		mRigidStaticContactUniqueIndices = mContactUniqueIndices + totalContacts;
		mArtiContactUniqueIndices = mRigidStaticContactUniqueIndices + totalStaticRigidContacts;
		mArtiStaticContactUniqueIndices = mArtiContactUniqueIndices + totalArticulationContacts;
		mArtiSelfContactUniqueIndices = mArtiStaticContactUniqueIndices + totalStaticArticulationContacts;	

		mArtiStaticConstraintStartIndex = reinterpret_cast<PxU32*>(mPinnedMemoryAllocator->allocate(allocationSizeArticulationCount, alignment));
		mArtiStaticConstraintCount = mArtiStaticConstraintStartIndex + mArticulationCount;
		mArtiStaticContactStartIndex = mArtiStaticConstraintCount + mArticulationCount;
		mArtiStaticContactCount = mArtiStaticContactStartIndex + mArticulationCount;

		mRigidStaticConstraintStartIndex = reinterpret_cast<PxU32*>(mPinnedMemoryAllocator->allocate(allocationSizeBodyCount, alignment));
		mRigidStaticConstraintCount = mRigidStaticConstraintStartIndex + mBodyCount;

		mGpuSolverCore->allocatePinnedDescriptors(*mPinnedMemoryAllocator);
	}

// PT: TODO: un-indent all of the above

void PxgGpuContext::doConstraintPrepGPU()
{
	PX_PROFILE_ZONE("GpuDynamics.ConstraintPrep", 0);
	/**
	* Things to do in here:
	* (1) constraint prep on GPU
	*/

	mGpuSolverCore->resetVelocities(mIsTGS);

	mGpuSolverCore->nonRigidConstraintPrepare(mArticulationCount);

	mGpuSolverCore->jointConstraintPrepareParallel(PxU32(mNum1dConstraintBatches + mNumStaticRigid1dConstraintBatches));
	mGpuSolverCore->contactConstraintPrepareParallel(PxU32(mNumContactBatches + mNumStaticRigidContactBatches));

	mGpuSolverCore->artiJointConstraintPrepare(PxU32(mNumArti1dConstraintBatches + mNumStaticArti1dConstraintBatches + mNumSelfArti1dConstraintBatches));
	mGpuSolverCore->artiContactConstraintPrepare(PxU32(mNumArtiContactBatches + mNumStaticArtiContactBatches + mNumSelfArtiContactBatches));

	mGpuArticulationCore->precomputeDependencies(PxMin(mIncrementalPartition.getNbPartitions(), mIncrementalPartition.getCombinedSlabMaxNbPartitions()));
}

void PxgGpuContext::doPreIntegrationGPU()
{
	const PxU32 offset = 1 + mKinematicCount;

	mGpuSolverCore->preIntegration(offset, mSolverBodyPool.size(), mDt, mGravity);

	if(mIsTGS)
		mIslandContextPool->mBiasCoefficient = PxMin(0.9f, 2.0f * PxSqrt(1.0f / mIslandContextPool->mNumPositionIterations));
}

void PxgGpuContext::doArticulationGPU()
{
	if(mIsTGS)
	{
		mGpuArticulationCore->computeUnconstrainedVelocities(mArticulationStartIndex, mArticulationCount, mDt, mGravity, 1.0f/mLengthScale, mIsExternalForcesEveryTgsIterationEnabled, mRecomputeArticulationBlockFormat);
	}
	else
	{
		mGpuArticulationCore->computeUnconstrainedVelocities(mArticulationStartIndex, mArticulationCount, mDt, mGravity, 1.0f/mLengthScale, false, mRecomputeArticulationBlockFormat);
		mGpuArticulationCore->setupInternalConstraints(mArticulationCount, mDt, mDt, 1.0f / mDt, false);
	}
}

void PxgGpuContext::doSoftbodyGPU()
{
	PxgSoftBodyCore* softBodyCore = static_cast<PxgSimulationController*>(mSimulationController)->getSoftBodyCore();
	if(softBodyCore)
		softBodyCore->updateTetraRotations();
}

void PxgGpuContext::doFEMClothGPU()
{
	// "I quickly checked, and it currently only resets Lagrange multiplier lambda used in the PBD framework.
	// For TGS, we don't use the Lagrange multiplier so no need to reset. Calling it on PGS only sounds okay to me."
	if(!mIsTGS)
	{
		PxgFEMClothCore* femClothCore = static_cast<PxgSimulationController*>(mSimulationController)->getFEMClothCore();
		if(femClothCore)
			femClothCore->preIteration();
	}
}

void PxgGpuContext::doConstraintPrePrepGPU()
{
	if(mIsTGS)
	{
		//Kick off articulation internal constraint setup code. At this point, we know the iteration count so we 
		//know how large time-steps will be.
		const PxReal stepDt = mDt / PxReal(mIslandContextPool->mNumPositionIterations);

		mGpuArticulationCore->setupInternalConstraints(mArticulationCount, stepDt, mDt, 1.0f / stepDt, true);
	}

	doConstraintPrePrepGPUCommon(mHasForceThresholds);
}

void PxgPostSolveTask::runInternal()
{
	nvtxRangePush("px:postSolveTask");
	mContext.doPostSolveTask(mCont);
	nvtxRangePop();
}

//This class kicks off constraint solve on GPU
void PxgGpuTask::runInternal()
{
	nvtxRangePush("px:gpuTask");
	mContext.mGpuSolverCore->acquireContext();

	mContext.doConstraintJointBlockPrePrepGPU();
	mContext.doConstraintPrepGPU();
	mContext.doConstraintSolveGPU(mMaxNodes, *mChangedHandleMap);

	mContext.mGpuSolverCore->releaseContext();
	nvtxRangePop();
}

void PxgGpuIntegrationTask::runInternal()
{
	nvtxRangePush("px:gpuIntegrationTask");
	mContext.mGpuSolverCore->acquireContext();

	//for articulation
	mContext.doArticulationGPU();

	//for soft body update rotation
	mContext.doSoftbodyGPU();

	//for FEM-cloth
	mContext.doFEMClothGPU();

	mContext.mGpuSolverCore->releaseContext();
	nvtxRangePop();
}

void PxgGpuPrePrepTask::runInternal()
{
	nvtxRangePush("px:gpuPrePrepTask");
	mContext.mGpuSolverCore->acquireContext();

	mContext.doPreIntegrationGPU();

	//for d6 joint
	mContext.doConstraintPrePrepGPU();

	PxgJointManager& jointManager = mContext.getSimulationController()->getJointManager();
	jointManager.reserveMemory(Dy::MAX_CONSTRAINT_ROWS);

	mContext.mGpuSolverCore->releaseContext();

	mContext.cpuJointPrePrepTask(mCont);
	nvtxRangePop();
}

void PxgGpuContext::updateBodyCore(PxBaseTask* continuation)
{
	mPostSolveTask.setContinuation(continuation);
	mPostSolveTask.removeReference();
}

//#define PXG_INCREMENTAL_SANITY_CHECKS
#if PX_ENABLE_ASSERTS
#ifdef PXG_INCREMENTAL_SANITY_CHECKS
	template <typename T>
	static bool noDuplicates(T* buffer, const PxU32 size)
	{
		for (PxU32 a = 0; a < size; ++a)
		{
			for (PxU32 b = 0; b < a; ++b)
			{
				if (buffer[a] == buffer[b])
					return false;
			}
		}
		return true;
	}
#else
	template <typename T>
	static bool noDuplicates(T*, const PxU32)
	{
		return true;
	}
#endif
#endif

static PX_FORCE_INLINE bool needsSolve(IG::IslandSim& islandSim, PxU32 bodyCount, PxU32 articulationCount)
{
	const PxU32 particleCount = islandSim.getNbActiveNodes(IG::Node::ePARTICLESYSTEM_TYPE);
	const PxU32 clothCount = islandSim.getNbActiveNodes(IG::Node::eDEFORMABLE_SURFACE_TYPE);
	const PxU32 softBodyCount = islandSim.getNbActiveNodes(IG::Node::eDEFORMABLE_VOLUME_TYPE);
	const bool needsSolve = (0 != bodyCount || 0 != articulationCount || particleCount || softBodyCount || clothCount);
	return needsSolve;
}

void PxgGpuContext::update(	Cm::FlushPool& flushPool, PxBaseTask* continuation, PxBaseTask* postPartitioningTask, PxBaseTask* /*lostTouchTask*/,
							PxvNphaseImplementationContext* nphase, PxU32 /*maxPatchesPerCM*/, PxU32 /*maxArticulationLinks*/, PxReal dt,
							const PxVec3& gravity, PxBitMapPinned& /*changedHandleMap*/)
{
	mGpuSolverCore->acquireContext();

	PxsContactManagerOutputIterator iterator = nphase->getContactManagerOutputs();
	PxsContactManagerOutput* gpuContactManagerOutputs = nphase->getGPUContactManagerOutputBase();

	mGPURestDistances = nphase->getGPURestDistances();
	mGPUShapeInteractions = nphase->getGPUShapeInteractions();
	mGPUTorsionalData = nphase->getGPUTorsionalData();

	mSolvedThisFrame = false;
	mOutputIterator = iterator;
	PX_ASSERT(noDuplicates(nphase->getLostFoundPatchManagers(), nphase->getNbLostFoundPatchManagers()));
	//First and foremost, we need to get a set of islands (bodies, constraints etc.)
	//These will be parameters
	IG::IslandSim& islandSim = mIslandManager.getAccurateIslandSim();

	const PxU32 bodyCount = islandSim.getNbActiveNodes(IG::Node::eRIGID_BODY_TYPE);
	const PxU32 articulationCount = islandSim.getNbActiveNodes(IG::Node::eARTICULATION_TYPE);

	mGpuSolverCore->setGpuContactManagerOutputBase(gpuContactManagerOutputs);

	if(!mIsTGS)
		mGpuSolverCore->syncSimulationController();	// PT: for some reason it's located here in PGS

	const PxU32 kinematicCount = islandSim.getNbActiveKinematics();
	mKinematicCount = kinematicCount;

	mArticulationCount = articulationCount;
	mArticulationStartIndex = 1 + kinematicCount + bodyCount;
	mRecomputeArticulationBlockFormat = getSimulationController()->getRecomputeArticulationBlockFormat();

	mBodyCount = bodyCount;

	// Auto-detect: no dynamic rigid bodies → all contacts are artic↔static
	// → CPU partition coloring is unnecessary (SOLVE_UNIFIED never launches)
	const bool staticOnly = (bodyCount == 0 && articulationCount > 0);
	mIncrementalPartition.setStaticContactsOnly(staticOnly);
	mGpuSolverCore->mStaticContactsOnly = staticOnly;

	mPinnedMemoryAllocator->reset();

#if PX_ENABLE_SIM_STATS
	mSimStats.mNbActiveKinematicBodies = islandSim.getNbActiveKinematics();
	mSimStats.mNbActiveDynamicBodies = islandSim.getNbActiveNodes(IG::Node::eRIGID_BODY_TYPE);
	mSimStats.mNbActiveConstraints = islandSim.getNbActiveEdges(IG::Edge::eCONSTRAINT);
	mSimStats.mNbPartitions = mIncrementalPartition.getNbPartitions();
#else
	PX_CATCH_UNDEFINED_ENABLE_SIM_STATS
#endif
	//mConstraintWriteBackStreamAllocator->reserve(sizeof(Dy::ConstraintWriteback) * nbConstraints);

	mConstraintsPerPartition.forceSize_Unsafe(0);
	mDt = dt;
	mInvDt = 1.f / dt;
	mGravity = gravity;
	//mEnableStabilization = enableStabilization;

	if(mIsTGS)
		mGpuSolverCore->syncSimulationController();
		
	{
		PX_PROFILE_ZONE("Dynamics.allocateBodyBuffers", 0);

		const PxU32 maxLinks = getSimulationController()->getSimulationCore()->getMaxArticulationLinks();
		const PxU32 maxDofs = getSimulationController()->getSimulationCore()->getMaxArticulationDofs();

		const PxU32 totalLinkJointRootStateByteSize = 
			PxgArticulationLinkJointRootStateData::computeStateDataBufferByteSizeAligned16(maxLinks, maxDofs, articulationCount);

		if (totalLinkJointRootStateByteSize > mLinkAndJointAndRootStateDataPool.capacity())
		{
			mLinkAndJointAndRootStateDataPool.forceSize_Unsafe(0);
			mLinkAndJointAndRootStateDataPool.reserve(totalLinkJointRootStateByteSize);
		}

		if (articulationCount > mArticulationSleepDataPool.capacity())
		{
			mArticulationSleepDataPool.forceSize_Unsafe(0);
			mArticulationSleepDataPool.reserve(articulationCount);
		}

		if (articulationCount*2 > mInternalResidualPerArticulationVelIter.capacity())
		{
			mInternalResidualPerArticulationVelIter.forceSize_Unsafe(0);
			mInternalResidualPerArticulationVelIter.reserve(articulationCount*2);
		}
		if (articulationCount*2 > mInternalResidualPerArticulationPosIter.capacity())
		{
			mInternalResidualPerArticulationPosIter.forceSize_Unsafe(0);
			mInternalResidualPerArticulationPosIter.reserve(articulationCount*2);
		}

		mLinkAndJointAndRootStateDataPool.forceSize_Unsafe(totalLinkJointRootStateByteSize);
		mArticulationSleepDataPool.forceSize_Unsafe(articulationCount);
		mInternalResidualPerArticulationVelIter.forceSize_Unsafe(articulationCount * 2);
		mInternalResidualPerArticulationPosIter.forceSize_Unsafe(articulationCount * 2);

		//1: Allocate buffers for all bodies (kinematic + dynamic)
		if ((kinematicCount + bodyCount + 1) > mSolverBodyPool.capacity())
		{
			//we don't need to dma up/back dynamic solver body data to gpu anymore. However, we still need to dma up static/kinematic solver body
			const PxU32 totalBodyAlignedCounts = (kinematicCount + bodyCount + 31 + 1) & (~31);

			mSolverBodyPool.forceSize_Unsafe(0);
			mSolverBodyPool.reserve(totalBodyAlignedCounts);

			mBody2WorldPool.forceSize_Unsafe(0);
			mBody2WorldPool.reserve(totalBodyAlignedCounts);

			mSolverBodyDataPool.forceSize_Unsafe(0);

			mSolverBodySleepDataPool.forceSize_Unsafe(0);
			mSolverBodySleepDataPool.reserve(totalBodyAlignedCounts);

			mSolverTxIDataPool.forceSize_Unsafe(0);
			mSolverTxIDataPool.reserve(totalBodyAlignedCounts);
		}

		if ((kinematicCount + bodyCount + 1 + articulationCount) > mActiveNodeIndex.capacity())
		{
			const PxU32 totalArticulationAlignedCounts = (kinematicCount + bodyCount + 1 + articulationCount + 31) & (~31);

			mActiveNodeIndex.forceSize_Unsafe(0);
			mActiveNodeIndex.reserve(totalArticulationAlignedCounts);
		}

		if ((kinematicCount + 31 + 1) > mSolverBodyDataPool.capacity())
		{
			mSolverBodyDataPool.reserve((kinematicCount + 31 + 1) & (~31));
		}

		mActiveNodeIndex.forceSize_Unsafe(1 + kinematicCount + bodyCount + articulationCount);

		//Set up constraint batches
		const PxU32 totalBodySize = 1 + kinematicCount + bodyCount;
		mSolverBodyPool.forceSize_Unsafe(totalBodySize);

		mBody2WorldPool.forceSize_Unsafe(totalBodySize);
		//we don't need to create dynamic solver body data in cpu anymore
		mSolverBodyDataPool.forceSize_Unsafe(1 + kinematicCount);
		//we need to dma up static+kinematic part of the sleepData and we dma up the whole sleepData array
		mSolverBodySleepDataPool.forceSize_Unsafe(totalBodySize);
		mSolverTxIDataPool.forceSize_Unsafe(totalBodySize);
	}

	if (getEnableDirectGPUAPI())
	{
		getSimulationController()->getJointManager().reserveMemoryPreAddRemove();
	}

	// Lean mode: skip task chain + island management, run GPU solver kernels directly.
	// All member variables (batch counts, partition data, island context) retain values from
	// the last normal (warmup) frame. The task chain reference counts work out because we
	// skip both mGpuTask.setContinuation(continuation) and updateIncrementalIslands(),
	// so neither continuation nor postPartitioningTask get the extra addRef.
	if (getSimulationController()->isLeanMode() && mIncrementalPartition.getStaticContactsOnly()
		&& needsSolve(islandSim, bodyCount, articulationCount))
	{
		const PxNodeIndex* const PX_RESTRICT articulationNodeIndices = islandSim.getActiveNodes(IG::Node::eARTICULATION_TYPE);

		PxMemCopy(mActiveNodeIndex.begin() + 1, islandSim.getActiveKinematics(), islandSim.getNbActiveKinematics() * sizeof(PxNodeIndex));
		PxMemCopy(mActiveNodeIndex.begin() + mArticulationStartIndex, articulationNodeIndices, sizeof(PxNodeIndex) * mArticulationCount);
		mActiveNodeIndex[0] = PxNodeIndex();

		PxgSimulationController* controller = static_cast<PxgSimulationController*>(mSimulationController);
		mGpuSolverCore->allocateSolverBodyBuffers(mIslandManager.getNbNodeHandles() + 1, mActiveNodeIndex, mArticulationCount, controller->getMaxLinks());

		mSolvedThisFrame = true;

		mSolverBodyPool[0] = mWorldSolverBody;
		mSolverBodyDataPool[0] = mWorldSolverBodyData;
		mSolverTxIDataPool[0] = mWorldTxIData;
		mSolverBodySleepDataPool[0] = mWorldSolverBodySleepData;

		simulateLean();

		mGpuSolverCore->releaseContext();
		return;
	}

	if (needsSolve(islandSim, bodyCount, articulationCount))
	{
		//Set up gpu workloads early!!!
		const PxNodeIndex* const PX_RESTRICT nodeIndices = islandSim.getActiveNodes(IG::Node::eRIGID_BODY_TYPE);
		const PxNodeIndex* const PX_RESTRICT articulationNodeIndices = islandSim.getActiveNodes(IG::Node::eARTICULATION_TYPE);

		PxMemCopy(mActiveNodeIndex.begin() + 1, islandSim.getActiveKinematics(), islandSim.getNbActiveKinematics() * sizeof(PxNodeIndex));
		PxMemCopy(mActiveNodeIndex.begin() + 1 + kinematicCount, nodeIndices, sizeof(PxNodeIndex) * mBodyCount);
		PxMemCopy(mActiveNodeIndex.begin() + mArticulationStartIndex, articulationNodeIndices, sizeof(PxNodeIndex) * mArticulationCount);

		mActiveNodeIndex[0] = PxNodeIndex();

		PxgSimulationController* controller = static_cast<PxgSimulationController*>(mSimulationController);
		const PxU32 maxLinks = controller->getMaxLinks();
		//DMA up the body data right now and any other data that might be available
		mGpuSolverCore->allocateSolverBodyBuffers(mIslandManager.getNbNodeHandles() + 1, mActiveNodeIndex, mArticulationCount, maxLinks);

		mSolvedThisFrame = true;

		//solver task chain!
		//Note - *all* work for *all* islands is processed in phases using a wide-model approach.
		//This is friendlier for the GPU but can be more wasteful in terms of memory
		mGpuTask.setContinuation(continuation);
		mGpuPrePrepTask.setContinuation(&mGpuTask);
		mPrepTask.setContinuation(&mGpuPrePrepTask);
		mPreIntegrationTask.setContinuation(&mPrepTask);
		mGpuIntegrationTask.setContinuation(&mGpuPrePrepTask);

		//Set up world rigid body
		mSolverBodyPool[0] = mWorldSolverBody;
		mSolverBodyDataPool[0] = mWorldSolverBodyData;
		mSolverTxIDataPool[0] = mWorldTxIData;
		mSolverBodySleepDataPool[0] = mWorldSolverBodySleepData;
			
		// these two are being launched immediately.
		mGpuIntegrationTask.removeReference();
		mPreIntegrationTask.removeReference();
	}

	// PT: when updateIncrementalIslands() is single-threaded this is a blocking call and we can use the
	// partitioning data when it returns. This is not the case anymore with multi-threaded implementations.

	// doConstraintPrePrepCommon() consumes the output of the incremental island building as part of mPrepTask
	mIncrementalPartition.updateIncrementalIslands(
		mIslandManager.getAccurateIslandSim(),
		mIslandManager.getAuxCpuData(),
		&flushPool, postPartitioningTask,
		mOutputIterator,	// PT: don't pass the local variable, it will go out of scope while the partitioning tasks are using it
		getSimulationController()->getBodySimManager(),
		getSimulationController()->getJointManager());

	// PT: all the code after the updateIncrementalIslands() call has been moved to PxgGpuContext::updatePostPartitioning() where
	// it can safely be executed after the potential updateIncrementalIslands() tasks are completed.

	mGpuSolverCore->releaseContext();
}

void PxgGpuContext::updatePostPartitioning(PxBaseTask* lostTouchTask, PxvNphaseImplementationContext* /*nphase*/,
	PxU32 maxPatchesPerCM, PxU32 /*maxArticulationLinks*/,
	PxReal /*dt*/, const PxVec3& /*gravity*/, PxBitMapPinned& changedHandleMap)
{
	// Lean mode: all solver work was done in update() via simulateLean().
	// Nothing to do here — member variables retain valid state from warmup.
	if (getSimulationController()->isLeanMode() && mIncrementalPartition.getStaticContactsOnly())
		return;

	mGpuSolverCore->acquireContext();

	IG::IslandSim& islandSim = mIslandManager.getAccurateIslandSim();

	const PxPinnedArray<PartitionIndexData>& partitionIndexDataIter = mIncrementalPartition.getPartitionIndexArray();
	const PxPinnedArray<PartitionNodeData>& partitionNodeData = mIncrementalPartition.getPartitionNodeArray();
	const PxPinnedArray<PxgSolverConstraintManagerConstants>& solverConstantData = mIncrementalPartition.getSolverConstants();
	const PxInt32ArrayPinned& partitionStartBatchIndexIter = mIncrementalPartition.getStartSlabPerPartition();
	const PxInt32ArrayPinned& partitionArticStartBatchIndexIter = mIncrementalPartition.getArticStartSlabPerPartition();
	const PxInt32ArrayPinned& partitionJointBatchCountIter = mIncrementalPartition.getNbJointsPerPartition();
	const PxInt32ArrayPinned& partitionArtiJointBatchCountIter = mIncrementalPartition.getNbArticJointsPerPartition();

	const PxArray<PxU32>& npIndexArrayIter = mIncrementalPartition.getNpIndexArray();
	PxInt32ArrayPinned& npIndexArrayStagingBuffer = mNodeIndicesStagingBuffer;
	PxInt32ArrayPinned& islandIds = mIslandIds;
	PxInt32ArrayPinned& islandStaticTouchCounts = mIslandStaticTouchCounts;

	const PxU32 nbConstraints = islandSim.getNbActiveEdges(IG::Edge::eCONSTRAINT);

	// At this point we are ready to allocate the pinned memory for the solver.
	allocateTempPinnedSolverMemoryCommon();

	const PxU32 bodyCount = mBodyCount;
	const PxU32 kinematicCount = mKinematicCount;
	const PxU32 articulationCount = mArticulationCount;

	//Force all bodies into a single island. The GPU partitioning provides better work balancing between blocks than just using multiple islands.
	PxgIslandContext& context = mIslandContextPool[0];
	context.mBodyStartIndex = 1 + kinematicCount;
	context.mBodyCount = bodyCount;
	context.mArticulationCount = articulationCount;
	context.mNumPositionIterations = context.mNumVelocityIterations = 0;
	mNumIslandContextPool = 1;

	//because updateIncrementalIslands add/remove joints based on activation 
	getSimulationController()->updateJointsAndSyncData();

	//reset number of frozen/unfrozen shapes to be zero
	mSimulationController->clear();

	PxgJointManager& jointManager = getSimulationController()->getJointManager();
	PX_ASSERT((jointManager.getCpuNbRigidConstraints() + jointManager.getCpuNbArtiConstraints() +
		jointManager.getGpuNbActiveRigidConstraints() + jointManager.getGpuNbActiveArtiConstraints()) == nbConstraints);

	PX_UNUSED(jointManager);

	const PxU32 nbPatches = mIncrementalPartition.getTotalContacts();	// PT: same as what mIncrementalPartition.updateIncrementalIslands() returned

#if PX_ENABLE_ASSERTS
	PxU32 accumulatedConstraints = mIncrementalPartition.getAccumulatedConstraintCount().size() == 0 ? 0 : mIncrementalPartition.getAccumulatedConstraintCount()[mIncrementalPartition.getAccumulatedConstraintCount().size() - 1];
	PxU32 accumulatedArtiConstraints = mIncrementalPartition.getAccumulatedArtiConstraintCount().size() == 0 ? 0 : mIncrementalPartition.getAccumulatedArtiConstraintCount()[mIncrementalPartition.getAccumulatedArtiConstraintCount().size() - 1];
	PX_ASSERT((nbPatches + islandSim.getNbActiveEdges(IG::Edge::eCONSTRAINT) + mIncrementalPartition.getTotalArticulationContacts()) == (accumulatedConstraints + accumulatedArtiConstraints + getSimulationController()->getBodySimManager().mTotalStaticArticJoints +
		getSimulationController()->getBodySimManager().mTotalSelfArticJoints + getSimulationController()->getBodySimManager().mTotalStaticRBJoints));
#endif

	{
		PX_PROFILE_ZONE("Dynamics.allocateConstraintBuffers", 0);

		//set the constraint batches number but we will do the actual memory allocation in doPartitionTask() method and free the excess amout in doConstraintPrePrepCommon(), so that
		//we can make sure mConstraintBatches is the last element allocated in the pinned memory allocator, therefore, we can shrunk the excess memory safely
		//mNumConstraintBatches = sentinel->constraints + sentinel->contactManagers;

		PxgBodySimManager& bodyManager = getSimulationController()->getBodySimManager();

		mNumContactManagers = nbPatches + bodyManager.mTotalStaticRBContacts;
		mNum1DConstraints = nbConstraints + bodyManager.mTotalStaticRBJoints;

		mThresholdStream->forceSize_Unsafe(0);
		mThresholdStream->reserve(PxNextPowerOfTwo(mNumContactManagers));

		mForceChangedThresholdStream->forceSize_Unsafe(0);
		mForceChangedThresholdStream->reserve(PxNextPowerOfTwo(mNumContactManagers));

		//Set up constraint batches
		//If there is no work to do then we can do nothing at all.

		// AD: this only works because we have the same if when setting up the task chain.
		// it's also in a somewhat weird place. We should analyze the dependencies, is all of the work we're doing up to here actually
		// required to happen even if we early-out here?
		if (!needsSolve(islandSim, bodyCount, articulationCount))
		{
			mGpuSolverCore->releaseContext();
			return;
		}

		//printf("NbarticBatches = %i, NbRigidBatches = %i\n", mIncrementalPartition.mNbArtiContactBatches, mIncrementalPartition.mNbContactBatches);
	}

	PxU32 descCount = 0;

	PxU32 currentDescIndex = 0;

	mGpuSolverCore->resetMemoryAllocator();

	PxU32 totalEdges = mIslandManager.getNbEdgeHandles();
	mTotalPreviousEdges = mTotalEdges;
	mTotalEdges = totalEdges;

	mGpuSolverCore->allocateFrictionPatchIndexStream(totalEdges * maxPatchesPerCM); //How many batches

	mGpuSolverCore->allocateFrictionCounts(totalEdges);

	currentDescIndex = mIncrementalPartition.getTotalConstraints() + mIncrementalPartition.getTotalContacts();

	context.mDescCount = currentDescIndex;
	context.mDescStartIndex = descCount;
	descCount += currentDescIndex;

	lostTouchTask->addReference();
	mLostTouchTask = lostTouchTask;

	npIndexArrayStagingBuffer.forceSize_Unsafe(0);
	npIndexArrayStagingBuffer.reserve(npIndexArrayIter.size());
	npIndexArrayStagingBuffer.forceSize_Unsafe(npIndexArrayIter.size());

	islandIds.forceSize_Unsafe(0);
	islandIds.reserve(islandSim.getNbNodes());
	islandIds.forceSize_Unsafe(islandSim.getNbNodes());

	islandStaticTouchCounts.forceSize_Unsafe(0);
	islandStaticTouchCounts.reserve(islandSim.getNbIslands());
	islandStaticTouchCounts.forceSize_Unsafe(islandSim.getNbIslands());

	//npIndexArray might be changed in island gen while solver is running, so we need to double buffer it
	PxMemCopy(npIndexArrayStagingBuffer.begin(), npIndexArrayIter.begin(), sizeof(PxU32) * npIndexArrayIter.size());
	PxMemCopy(islandIds.begin(), islandSim.getIslandIds(), sizeof(PxU32) * islandSim.getNbNodes());
	PxMemCopy(islandStaticTouchCounts.begin(), islandSim.getIslandStaticTouchCount(), sizeof(PxU32) * islandSim.getNbIslands());

	const PxInt32ArrayPinned& nodeInteractions = mIncrementalPartition.getNodeInteractionCountArray();

	// Phase B: Build GPU contact lists + skip contact stream H→D
	if (mIncrementalPartition.getStaticContactsOnly())
	{
		CUstream solverStream = mGpuSolverCore->getStream();
		if (getNarrowphaseCore()->mCudaContext->isSingleStreamMode())
			launchBuildContactMappingGPU(solverStream);  // GPU-only, no H2D
		else
			buildAndUploadContactMapping(solverStream);  // CPU path with H2D (warmup)
		launchBuildStaticContactLists(solverStream);

		// GPU-direct descriptor patch: update variable count fields from GPU kernel output.
		// mStaticUniqueIdCounter_d has the actual contact count (set by buildStaticContactLists kernel).
		// Patch PxgConstraintPrepareDesc.numArtiStaticContactBatches + totalCurrentEdges on device.
		if (getNarrowphaseCore()->mCudaContext->isSingleStreamMode())
		{
			PxCudaContext* cudaCtx = getNarrowphaseCore()->mCudaContext;
			CUdeviceptr prepDescd = mGpuSolverCore->getPrepDescDeviceptr();
			if (prepDescd)
			{
				cudaCtx->memcpyDtoDAsync(
					prepDescd + offsetof(PxgConstraintPrepareDesc, numArtiStaticContactBatches),
					mStaticUniqueIdCounter_d, sizeof(PxU32), solverStream);
				cudaCtx->memcpyDtoDAsync(
					prepDescd + offsetof(PxgConstraintPrepareDesc, totalCurrentEdges),
					mStaticUniqueIdCounter_d, sizeof(PxU32), solverStream);
			}

		}
	}

	// Phase B+: skip H2D upload entirely when GPU kernel already built contact data on device.
	// gpuMemDMAUpContactData uploads partition/contact data from pinned memory → device,
	// but in mStaticContactsOnly mode, the GPU kernel wrote directly to device and
	// the pointer override below (line ~2968) replaces all uploaded pointers anyway.
	if (!mIncrementalPartition.getStaticContactsOnly())
	{
		mGpuSolverCore->gpuMemDMAUpContactData(mContactStreamAllocators[mCurrentContactStream],
			PxToU32(mContactStreamPool.mSharedDataIndex),
			mContactStreamPool.mSharedDataIndexGPU,
			mPatchStreamAllocators[mCurrentContactStream],
			PxToU32(mPatchStreamPool.mSharedDataIndex),
			mPatchStreamPool.mSharedDataIndexGPU,
			mNumContactManagers,
			partitionIndexDataIter.begin(), partitionNodeData.begin(), solverConstantData.begin(), solverConstantData.size(), partitionIndexDataIter.size(),
			partitionStartBatchIndexIter.begin(), partitionArticStartBatchIndexIter.begin(), partitionJointBatchCountIter.begin(), partitionArtiJointBatchCountIter.begin(),
			partitionStartBatchIndexIter.size(),
			mIncrementalPartition.getDestroyedContactEdgeIndices().begin(), mIncrementalPartition.getDestroyedContactEdgeIndices().size(),
			npIndexArrayStagingBuffer.begin(), npIndexArrayStagingBuffer.size(),
			/*jointManager.mGpuJointData, jointManager.mGpuJointPrePrep, gpuJointSize,*/ mConstraintWriteBackPool.size(),
			islandIds.begin(), nodeInteractions.begin(), islandIds.size(), islandStaticTouchCounts.begin(), islandStaticTouchCounts.size());
	}

	// Phase B: override partition data pointers to GPU kernel device buffers
	if (mIncrementalPartition.getStaticContactsOnly())
	{
		PxU32 n = mStaticContactMappingCount;
		mGpuSolverCore->mPartitionNodeData.set(mStaticNodeArray_d, n * sizeof(PartitionNodeData));
		mGpuSolverCore->mSolverConstantData.set(mStaticSolverConstants_d, n * sizeof(PxgSolverConstraintManagerConstants));
		mGpuSolverCore->mPartitionIndexData.set(mStaticPartIndexArray_d, n * sizeof(PartitionIndexData));
		mGpuSolverCore->mNpIndexArray.set(mStaticNpIndexArray_d, n * sizeof(PxU32));
	}

	mGpuSolverCore->releaseContext();

	mGpuTask.setMaxNodesAndWordCounts(mIslandManager.getNbNodeHandles(), changedHandleMap);

	//Now we have kicked off all the atom integration and pre-prep work, so we can permit the remaining phases of the solver to run...
	//mPostSolveTask.removeReference();
	mGpuTask.removeReference();
	mGpuPrePrepTask.removeReference();
	mPrepTask.removeReference();
}

}
