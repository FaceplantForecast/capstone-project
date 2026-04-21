/**
 *   @file  mmwave_fullcfg.c
 *
 *   @brief
 *      The file implements the functions which are required to support
 *      the FULL configuration mode.
 *
 *  \par
 *  NOTE:
 *      (C) Copyright 2016-2025 Texas Instruments, Inc.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *    Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 *    Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the
 *    distribution.
 *
 *    Neither the name of Texas Instruments Incorporated nor the names of
 *    its contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**************************************************************************
 *************************** Include Files ********************************
 **************************************************************************/
#include <stdint.h>
#include <string.h>
#include <C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mmwave_mcuplus_sdk_04_07_01_04/ti/common/syscommon.h>

/* Includes from MCU Plus SDK */
#include <kernel/dpl/DebugP.h>
#include <kernel/dpl/HeapP.h>

/* Includes from Mmwave SDK */
#include <C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mmwave_mcuplus_sdk_04_07_01_04/ti/control/mmwave/mmwave.h>
#include <C:/ti/mmwave_mcuplus_sdk_04_07_01_04/mmwave_mcuplus_sdk_04_07_01_04/ti/control/mmwave/include/mmwave_internal.h>

/* User defined heap memory and handle */
#define MMWAVE_HEAP_MEM_SIZE  (8*1024u)
static uint8_t gMmwHeapMem[MMWAVE_HEAP_MEM_SIZE] __attribute__((aligned(HeapP_BYTE_ALIGNMENT))) = {0};
static HeapP_Object gMmwHeapObj;

static bool isHeapMemAllocated = false;

/**************************************************************************
 **************************** Local Functions *****************************
 **************************************************************************/
static void MMWave_internalDelChirp (MMWave_Profile* ptrMMWaveProfile, MMWave_Chirp* ptrMMWaveChirp);
static void MMWave_internalDelProfile (MMWave_MCB* ptrMMWaveMCB, MMWave_Profile* ptrMMWaveProfile);
static void MMWave_internalDelBpmChirp (MMWave_MCB* ptrMMWaveMCB, MMWave_BpmChirp* ptrMMWaveBpmChirp);
static void MMWave_internalDelPhaseShiftChirp (MMWave_MCB* ptrMMWaveMCB, MMWave_phaseShiftChirp* ptrMMWavePhaseShiftChirp);
static void MMWave_internalDelAdvChirp(MMWave_MCB* ptrMMWaveMCB, MMWave_AdvChirp* ptrMMWaveAdvChirp);
static void MMWave_internalDelAdvChirpLUTData(rlInt32_t* ptrMMWaveLUTData);
/**************************************************************************
 ******************* mmWave Configuration Functions ***********************
 **************************************************************************/

/**
 *  @b Description
 *  @n
 *      This is an *internal* function which is used to delete the chirp
 *      associated with a profile. The function does not hold the critical
 *      section.
 *
 *  @param[in]  ptrMMWaveProfile
 *      Pointer to the profile
 *  @param[in]  ptrMMWaveChirp
 *      Pointer to the chirp to be deleted
 *
 *  \ingroup MMWAVE_CTRL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Not applicable
 */
static void MMWave_internalDelChirp
(
    MMWave_Profile* ptrMMWaveProfile,
    MMWave_Chirp*   ptrMMWaveChirp
)
{
    /* Remove the chirp from the profile. */
    MMWave_listRemoveNode ((MMWave_ListNode**)&ptrMMWaveProfile->ptrChirpList, (MMWave_ListNode*)ptrMMWaveChirp);

    /* Decrement the number of chirps which are linked to the profile: */
    ptrMMWaveProfile->numChirps--;

    /* Cleanup the chirp memory: */
    HeapP_free(&gMmwHeapObj, (void *)ptrMMWaveChirp);
}

/**
 *  @b Description
 *  @n
 *      The function is used to delete the chirp from an existing profile.
 *
 *  @param[in]  profileHandle
 *      Handle to the profile to which the chirp is to be deleted
 *  @param[in]  chirpHandle
 *      Handle to the chirp to be deleted
 *  @param[out] errCode
 *      Encoded Error code populated by the API on an error
 *
 *  \ingroup MMWAVE_CTRL_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MMWave_delChirp
(
    MMWave_ProfileHandle    profileHandle,
    MMWave_ChirpHandle      chirpHandle,
    int32_t*                errCode
)
{
    MMWave_Chirp*       ptrMMWaveChirp;
    MMWave_Profile*     ptrMMWaveProfile;
    int32_t             retVal = MINUS_ONE;

    /* Initialize the error code: */
    *errCode = 0;

    /* Sanity Check: Validate the arguments */
    if ((profileHandle == NULL) || (chirpHandle == NULL))
    {
        /* Error: Invalid arguments */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
        goto exit;
    }

    /* Get the profile & chirp handle: */
    ptrMMWaveProfile = (MMWave_Profile*)profileHandle;
    ptrMMWaveChirp   = (MMWave_Chirp*)chirpHandle;

    /* Sanity Check: Each profile is linked to the mmWave module */
    DebugP_assert (ptrMMWaveProfile->ptrMMWaveMCB != NULL);

    /* Sanity Check: These API are available only in FULL configuration mode */
    DebugP_assert (ptrMMWaveProfile->ptrMMWaveMCB->initCfg.cfgMode == MMWave_ConfigurationMode_FULL);

    /* Sanity Check: Are there any chirps to be deleted */
    if (ptrMMWaveProfile->numChirps == 0U)
    {
        /* Error: There are no chirps to delete. This is an invalid usage from the application */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
        goto exit;
    }

    /* Critical Section Enter: Protect the 'Chirp List' */
    SemaphoreP_pend (&(ptrMMWaveProfile->ptrMMWaveMCB->cfgSemHandle), SystemP_WAIT_FOREVER);

    /* Delete the chirp: */
    MMWave_internalDelChirp (ptrMMWaveProfile, ptrMMWaveChirp);

    /* Critical Section Exit: */
    SemaphoreP_post (&(ptrMMWaveProfile->ptrMMWaveMCB->cfgSemHandle));

    /* Setup the return value: */
    retVal = 0;

exit:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to add the chirp to an existing profile.
 *
 *  @param[in]  profileHandle
 *      Handle to the profile to which the chirp is to be added
 *  @param[in]  ptrChirpCfg
 *      Pointer to the chirp configuration
 *  @param[out] errCode
 *      Encoded Error code populated by the API on an error
 *
 *  \ingroup MMWAVE_CTRL_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   Handle to the chirp
 *  @retval
 *      Error   -   NULL
 */
MMWave_ChirpHandle MMWave_addChirp
(
    MMWave_ProfileHandle    profileHandle,
    const rlChirpCfg_t*     ptrChirpCfg,
    int32_t*                errCode
)
{
    MMWave_Chirp*           ptrMMWaveChirp;
    MMWave_Profile*         ptrMMWaveProfile;
    MMWave_ChirpHandle      retHandle = NULL;

    /* Initialize the error code: */
    *errCode = 0;

    /* Sanity Check: Validate the arguments */
    if ((profileHandle == NULL) || (ptrChirpCfg == NULL))
    {
        /* Error: Invalid argument */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
        goto exit;
    }

    /* Get the profile handle: */
    ptrMMWaveProfile = (MMWave_Profile*)profileHandle;

    /* Sanity Check: Each profile is linked to the mmWave module */
    DebugP_assert (ptrMMWaveProfile->ptrMMWaveMCB != NULL);

    /* Sanity Check: These API are available only in FULL configuration mode */
    DebugP_assert (ptrMMWaveProfile->ptrMMWaveMCB->initCfg.cfgMode == MMWave_ConfigurationMode_FULL);

    /* Allocate memory for the chirp: */
    ptrMMWaveChirp = (MMWave_Chirp*) HeapP_alloc(&gMmwHeapObj, (sizeof(MMWave_Chirp)));
    if (ptrMMWaveChirp == NULL)
    {
        /* Error: Out of memory */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_ENOMEM, 0);
        goto exit;
    }

    /* Initialize the allocated memory for the chirp: */
    memset ((void*)ptrMMWaveChirp, 0, sizeof(MMWave_Chirp));

    /* Populate the Chirp: */
    memcpy ((void *)&ptrMMWaveChirp->chirpCfg, (const void*)ptrChirpCfg, sizeof(rlChirpCfg_t));
    ptrMMWaveChirp->ptrMMWaveProfile = ptrMMWaveProfile;

    /* Critical Section Enter: Protect the 'Chirp List' */
    SemaphoreP_pend (&(ptrMMWaveProfile->ptrMMWaveMCB->cfgSemHandle), SystemP_WAIT_FOREVER);

    /* Add the chirp to the profile list */
    MMWave_listCat ((MMWave_ListNode**)&ptrMMWaveProfile->ptrChirpList, (MMWave_ListNode**)&ptrMMWaveChirp);

    /* Increment the number of chirps which are linked to the profile: */
    ptrMMWaveProfile->numChirps++;

    /* Critical Section Exit: */
    SemaphoreP_post (&(ptrMMWaveProfile->ptrMMWaveMCB->cfgSemHandle));

    /* Setup the return handle: */
    retHandle = (MMWave_ChirpHandle)ptrMMWaveChirp;

exit:
    return retHandle;
}

/**
 *  @b Description
 *  @n
 *      The function is used to get the chirp configuration given
 *      the chirp handle
 *
 *  @param[in]  chirpHandle
 *      Handle to the chirp
 *  @param[out] ptrChirpCfg
 *      Pointer to the chirp configuration populated by the API
 *  @param[out] errCode
 *      Encoded Error code populated by the API on an error
 *
 *  \ingroup MMWAVE_CTRL_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MMWave_getChirpCfg
(
    MMWave_ChirpHandle  chirpHandle,
    rlChirpCfg_t*       ptrChirpCfg,
    int32_t*            errCode
)
{
    MMWave_Chirp*   ptrMMWaveChirp;
    int32_t         retVal = MINUS_ONE;

    /* Initialize the error code: */
    *errCode = 0;

    /* Sanity Check: Validate the arguments */
    if ((chirpHandle == NULL) || (ptrChirpCfg == NULL))
    {
        /* Error: Invalid arguments */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
        goto exit;
    }

    /* Get the pointer to the Chirp: */
    ptrMMWaveChirp = (MMWave_Chirp*)chirpHandle;

    /* Chirps are always linked to profiles and profiles to the mmWave control module. */
    DebugP_assert (ptrMMWaveChirp->ptrMMWaveProfile != NULL);
    DebugP_assert (ptrMMWaveChirp->ptrMMWaveProfile->ptrMMWaveMCB != NULL);

    /* Sanity Check: These API are available only in FULL configuration mode */
    DebugP_assert (ptrMMWaveChirp->ptrMMWaveProfile->ptrMMWaveMCB->initCfg.cfgMode == MMWave_ConfigurationMode_FULL);

    /* Copy over the configuration: */
    memcpy ((void*)ptrChirpCfg, (void*)&ptrMMWaveChirp->chirpCfg, sizeof(rlChirpCfg_t));

    /* Setup the return value: */
    retVal = 0;

exit:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to get the number of chirps attached to a profile
 *
 *  @param[in]  profileHandle
 *      Handle to the profile
 *  @param[out] numChirps
 *      Number of chirps attached to a profile
 *  @param[out] errCode
 *      Encoded Error code populated by the API on an error
 *
 *  \ingroup MMWAVE_CTRL_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MMWave_getNumChirps
(
    MMWave_ProfileHandle    profileHandle,
    uint32_t*               numChirps,
    int32_t*                errCode
)
{
    MMWave_Profile*     ptrMMWaveProfile;
    int32_t             retVal = MINUS_ONE;

    /* Initialize the error code: */
    *errCode = 0;

    /* Sanity Check: Validate the arguments */
    if ((profileHandle == NULL) || (numChirps == NULL))
    {
        /* Error: Invalid arguments */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
        goto exit;
    }

    /* Get the pointer to the Profile: */
    ptrMMWaveProfile = (MMWave_Profile*)profileHandle;

    /* Sanity Check: Profiles are always linked to the control module */
    DebugP_assert (ptrMMWaveProfile->ptrMMWaveMCB != NULL);

    /* Sanity Check: These API are available only in FULL configuration mode */
    DebugP_assert (ptrMMWaveProfile->ptrMMWaveMCB->initCfg.cfgMode == MMWave_ConfigurationMode_FULL);

    /* Get the number of chirps: */
    *numChirps = ptrMMWaveProfile->numChirps;

    /* Setup the return value: */
    retVal = 0;

exit:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function can be used by the application to get the chirp handle
 *      at the specified index. If the index exceeds the number of chirps
 *      configured the function will fail with the error code.
 *
 *  @param[in]  profileHandle
 *      Handle to the profile
 *  @param[in]  chirpIndex
 *      Chirp Index for which the handle is needed. Set to 1 to get the
 *      first chirp index etc
 *  @param[out] chirpHandle
 *      Populated chirp handle
 *  @param[out] errCode
 *      Encoded Error code populated by the API on an error
 *
 *  \ingroup MMWAVE_CTRL_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MMWave_getChirpHandle
(
    MMWave_ProfileHandle    profileHandle,
    uint32_t                chirpIndex,
    MMWave_ChirpHandle*     chirpHandle,
    int32_t*                errCode
)
{
    MMWave_Profile*     ptrMMWaveProfile;
    MMWave_Chirp*       ptrMMWaveChirp;
    uint32_t            index  = 1U;
    int32_t             retVal = MINUS_ONE;
    int32_t             endProcessing = 0;

    /* Initialize the error code: */
    *errCode     = 0;

    /* Sanity Check: Validate the arguments */
    if ((profileHandle == NULL) || (chirpHandle == NULL))
    {
        /* Error: Invalid arguments */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
        goto exit;
    }

    /* Get the pointer to the Profile: */
    ptrMMWaveProfile = (MMWave_Profile*)profileHandle;

    /* Sanity Check: Profiles are always linked to the control module */
    DebugP_assert (ptrMMWaveProfile->ptrMMWaveMCB != NULL);

    /* Sanity Check: These API are available only in FULL configuration mode */
    DebugP_assert (ptrMMWaveProfile->ptrMMWaveMCB->initCfg.cfgMode == MMWave_ConfigurationMode_FULL);

    /* Initialize the chirp handle */
    *chirpHandle = NULL;

    /* Critical Section Enter: Protect the 'Chirp List' */
    SemaphoreP_pend (&(ptrMMWaveProfile->ptrMMWaveMCB->cfgSemHandle), SystemP_WAIT_FOREVER);

    /* Get the head of the chirp list: */
    ptrMMWaveChirp = (MMWave_Chirp*)MMWave_listGetHead ((MMWave_ListNode**)&ptrMMWaveProfile->ptrChirpList);
    while (endProcessing == 0)
    {
        /* Have we reached the end of the list? */
        if (ptrMMWaveChirp == NULL)
        {
            /* YES: Control comes here indicates that the chirp index specified exceeds the
             * configured number of chirps. We are done with the processing */
            *errCode      = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
            endProcessing = 1;
        }
        else
        {
            /* Is this what we are looking for? */
            if (index == chirpIndex)
            {
                /* YES: Setup the chirp handle. */
                *chirpHandle  = (MMWave_ChirpHandle)ptrMMWaveChirp;
                retVal        = 0;
                endProcessing = 1;
            }

            /* Get the next element: */
            index = index + 1U;
            ptrMMWaveChirp = (MMWave_Chirp*)MMWave_listGetNext ((MMWave_ListNode*)ptrMMWaveChirp);
        }
    }

    /* Critical Section Exit: */
    SemaphoreP_post (&(ptrMMWaveProfile->ptrMMWaveMCB->cfgSemHandle));

exit:
    return retVal;
}


/**
 *  @b Description
 *  @n
 *      The function is used to add the profile with the specific
 *      profile configuration.
 *
 *  @param[in]  mmWaveHandle
 *      Handle to the mmWave module
 *  @param[in]  ptrProfileCfg
 *      Pointer to the profile configuration
 *  @param[out] errCode
 *      Encoded Error code populated by the API on an error
 *
 *  \ingroup MMWAVE_CTRL_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   Handle to the profile
 *  @retval
 *      Error   -   NULL
 */
MMWave_ProfileHandle MMWave_addProfile
(
    MMWave_Handle           mmWaveHandle,
    const rlProfileCfg_t*   ptrProfileCfg,
    int32_t*                errCode
)
{
    MMWave_MCB*             ptrMMWaveMCB;
    MMWave_Profile*         ptrMMWaveProfile;
    MMWave_ProfileHandle    retHandle = NULL;

    /* Initialize the error code: */
    *errCode = 0;

    if(!isHeapMemAllocated)
    {
        /* create heap for profile, chirp and BPM config. */
        HeapP_construct(&gMmwHeapObj, gMmwHeapMem, MMWAVE_HEAP_MEM_SIZE);

        isHeapMemAllocated = true;
    }

    /* Sanity Check: Validate the arguments */
    if ((mmWaveHandle == NULL) || (ptrProfileCfg == NULL))
    {
        /* Error: Invalid arguments */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
        goto exit;
    }

    /* Get the pointer to the mmWave MCB: */
    ptrMMWaveMCB = (MMWave_MCB*)mmWaveHandle;

    /* Sanity Check: These API are available only in FULL configuration mode */
    DebugP_assert (ptrMMWaveMCB->initCfg.cfgMode == MMWave_ConfigurationMode_FULL);

    /* Allocate memory for the Profile: */
    ptrMMWaveProfile = HeapP_alloc(&gMmwHeapObj, (sizeof(MMWave_Profile)));

    if (ptrMMWaveProfile == NULL)
    {
        /* Error: Out of memory */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_ENOMEM, 0);
        goto exit;
    }

    /* Initialize the allocated memory: */
    memset ((void *)ptrMMWaveProfile, 0, sizeof(MMWave_Profile));

    /* Populate the profile: */
    memcpy ((void*)&ptrMMWaveProfile->profileCfg, (const void*)ptrProfileCfg, sizeof(rlProfileCfg_t));
    ptrMMWaveProfile->ptrMMWaveMCB = ptrMMWaveMCB;

    /* Critical Section Enter: Protect the 'Profile List' */
    SemaphoreP_pend (&(ptrMMWaveMCB->cfgSemHandle), SystemP_WAIT_FOREVER);

    /* Add the profile to the Profile List  */
    MMWave_listAdd ((MMWave_ListNode**)&ptrMMWaveMCB->ptrProfileList, (MMWave_ListNode*)ptrMMWaveProfile);

    /* Critical Section Exit: */
    SemaphoreP_post (&(ptrMMWaveMCB->cfgSemHandle));

    /* Profile has been successfully registered */
    retHandle = (MMWave_ProfileHandle)ptrMMWaveProfile;

exit:
    return retHandle;
}

/**
 *  @b Description
 *  @n
 *      This is an *internal* function which is used to delete the profile
 *      The function does not hold the critical section.
 *
 *  @param[in]  ptrMMWaveMCB
 *      Pointer to the mmWave control module
 *  @param[in]  ptrMMWaveProfile
 *      Pointer to the profile to be deleted
 *
 *  \ingroup MMWAVE_CTRL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Not applicable
 */
static void MMWave_internalDelProfile (MMWave_MCB* ptrMMWaveMCB, MMWave_Profile* ptrMMWaveProfile)
{
    /* Remove the profile from the Profile List  */
    MMWave_listRemoveNode ((MMWave_ListNode**)&ptrMMWaveMCB->ptrProfileList, (MMWave_ListNode*)ptrMMWaveProfile);

    /* Cleanup the profile memory: */
    HeapP_free(&gMmwHeapObj, (void *)ptrMMWaveProfile);
    return;
}

/**
 *  @b Description
 *  @n
 *      The function is used to delete the profile. This will also delete all
 *      the chirps which are still attached to the profile.
 *
 *  @param[in]  mmWaveHandle
 *      Handle to the mmWave module
 *  @param[in]  profileHandle
 *      Handle to the profile to be deleted
 *  @param[out] errCode
 *      Encoded Error code populated by the API on an error
 *
 *  \ingroup MMWAVE_CTRL_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MMWave_delProfile
(
    MMWave_Handle           mmWaveHandle,
    MMWave_ProfileHandle    profileHandle,
    int32_t*                errCode
)
{
    MMWave_MCB*             ptrMMWaveMCB;
    MMWave_Profile*         ptrMMWaveProfile;
    MMWave_Chirp*           ptrMMWaveChirp;
    int32_t                 retVal = MINUS_ONE;

    /* Initialize the error code: */
    *errCode = 0;

    /* Sanity Check: Validate the arguments */
    if ((mmWaveHandle == NULL) || (profileHandle == NULL))
    {
        /* Error: Invalid arguments */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
        goto exit;
    }

    /* Setup the pointers: */
    ptrMMWaveMCB     = (MMWave_MCB*)mmWaveHandle;
    ptrMMWaveProfile = (MMWave_Profile*)profileHandle;

    /* Sanity Check: These API are available only in FULL configuration mode */
    DebugP_assert (ptrMMWaveMCB->initCfg.cfgMode == MMWave_ConfigurationMode_FULL);

    /* Critical Section Enter: Protect the 'Profile & Chirp List' */
    SemaphoreP_pend (&(ptrMMWaveMCB->cfgSemHandle), SystemP_WAIT_FOREVER);

    /* Cycle through all the registered chirps: */
    ptrMMWaveChirp = (MMWave_Chirp*)MMWave_listGetHead ((MMWave_ListNode**)&ptrMMWaveProfile->ptrChirpList);
    while (ptrMMWaveChirp != NULL)
    {
        /* Delete the chirp: Use the internal API since the semaphore is already held. */
        MMWave_internalDelChirp (ptrMMWaveProfile, ptrMMWaveChirp);

        /* Cycle through the list again and get the new head. */
        ptrMMWaveChirp = (MMWave_Chirp*)MMWave_listGetHead ((MMWave_ListNode**)&ptrMMWaveProfile->ptrChirpList);
    }

    /* Delete the profile: */
    MMWave_internalDelProfile (ptrMMWaveMCB, ptrMMWaveProfile);

    /* Critical Section Exit: */
    SemaphoreP_post (&(ptrMMWaveMCB->cfgSemHandle));

    /* Setup the return value: */
    retVal = 0;

exit:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to get the number of profiles which have been added.
 *
 *  @param[in]  mmWaveHandle
 *      Handle to the mmWave module
 *  @param[out] numProfiles
 *      Number of added profiles populated by the API
 *  @param[out] errCode
 *      Encoded Error code populated by the API on an error
 *
 *  \ingroup MMWAVE_CTRL_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MMWave_getNumProfiles(MMWave_Handle mmWaveHandle, uint32_t* numProfiles, int32_t* errCode)
{
    MMWave_MCB*         ptrMMWaveMCB;
    MMWave_Profile*     ptrMMWaveProfile;
    int32_t             retVal = MINUS_ONE;

    /* Initialize the error code: */
    *errCode = 0;

    /* Sanity Check: Validate the arguments */
    if ((mmWaveHandle == NULL) || (numProfiles == NULL))
    {
        /* Error: Invalid arguments */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
        goto exit;
    }

    /* Get the pointer to the mmWave control module: */
    ptrMMWaveMCB = (MMWave_MCB*)mmWaveHandle;

    /* Sanity Check: These API are available only in FULL configuration mode */
    DebugP_assert (ptrMMWaveMCB->initCfg.cfgMode == MMWave_ConfigurationMode_FULL);

    /* Critical Section Enter: Protect the 'Profile List' */
    SemaphoreP_pend (&(ptrMMWaveMCB->cfgSemHandle), SystemP_WAIT_FOREVER);

    /* Initialize the number of profiles */
    *numProfiles = 0U;

    /* Cycle through the profile list */
    ptrMMWaveProfile = (MMWave_Profile*)MMWave_listGetHead ((MMWave_ListNode**)&ptrMMWaveMCB->ptrProfileList);
    while (ptrMMWaveProfile != NULL)
    {
        /* Increment the number of profiles */
        *numProfiles = *numProfiles + 1U;
        ptrMMWaveProfile = (MMWave_Profile*)MMWave_listGetNext ((MMWave_ListNode*)ptrMMWaveProfile);
    }

    /* Critical Section Exit: */
    SemaphoreP_post (&(ptrMMWaveMCB->cfgSemHandle));

    /* Setup the return value: */
    retVal = 0;

exit:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to get the profile handle for the specific profile
 *      identifier.
 *
 *  @param[in]  mmWaveHandle
 *      Handle to the mmWave module
 *  @param[in]  profileId
 *      Profile Id
 *  @param[out] profileHandle
 *      Handle to the profile populated by the API
 *  @param[out] errCode
 *      Encoded Error code populated by the API on an error
 *
 *  \ingroup MMWAVE_CTRL_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MMWave_getProfileHandle
(
    MMWave_Handle           mmWaveHandle,
    uint8_t                 profileId,
    MMWave_ProfileHandle*   profileHandle,
    int32_t*                errCode
)
{
    MMWave_MCB*         ptrMMWaveMCB;
    MMWave_Profile*     ptrMMWaveProfile;
    int32_t             retVal = MINUS_ONE;

    /* Initialize the error code: */
    *errCode = 0;

    /* Sanity Check: Validate the arguments */
    if ((mmWaveHandle == NULL) || (profileHandle == NULL))
    {
        /* Error: Invalid arguments */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
        goto exit;
    }

    /* Setup the pointers: */
    ptrMMWaveMCB = (MMWave_MCB*)mmWaveHandle;

    /* Sanity Check: These API are available only in FULL configuration mode */
    DebugP_assert (ptrMMWaveMCB->initCfg.cfgMode == MMWave_ConfigurationMode_FULL);

    /* Critical Section Enter: Protect the 'Profile List' */
    SemaphoreP_pend (&(ptrMMWaveMCB->cfgSemHandle), SystemP_WAIT_FOREVER);

    /* Cycle through the profile list */
    ptrMMWaveProfile = (MMWave_Profile*)MMWave_listGetHead ((MMWave_ListNode**)&ptrMMWaveMCB->ptrProfileList);
    while (ptrMMWaveProfile != NULL)
    {
        /* Is this what we are looking for? */
        if (profileId == ptrMMWaveProfile->profileCfg.profileId)
        {
            /* YES: Setup the profile handle */
            *profileHandle = (MMWave_ProfileHandle)ptrMMWaveProfile;
            retVal = 0;
            break;
        }

        /* Get the next element: */
        ptrMMWaveProfile = (MMWave_Profile*)MMWave_listGetNext ((MMWave_ListNode*)ptrMMWaveProfile);
    }

    /* Critical Section Exit: */
    SemaphoreP_post (&(ptrMMWaveMCB->cfgSemHandle));

    /* Did we find a match? */
    if (ptrMMWaveProfile == NULL)
    {
        /* Error: No matching profile identifier found. Setup the error code. */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_ENOTFOUND, 0);
    }

exit:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to get the chirp configuration given
 *      the chirp handle
 *
 *  @param[in]  profileHandle
 *      Handle to the profile
 *  @param[out] ptrProfileCfg
 *      Pointer to the profile configuration populated by the API
 *  @param[out] errCode
 *      Encoded Error code populated by the API on an error
 *
 *  \ingroup MMWAVE_CTRL_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MMWave_getProfileCfg
(
    MMWave_ProfileHandle    profileHandle,
    rlProfileCfg_t*         ptrProfileCfg,
    int32_t*                errCode
)
{
    MMWave_Profile*     ptrMMWaveProfile;
    int32_t             retVal = MINUS_ONE;

    /* Initialize the error code: */
    *errCode = 0;

    /* Sanity Check: Validate the arguments */
    if ((profileHandle == NULL) || (ptrProfileCfg == NULL))
    {
        /* Error: Invalid arguments */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
        goto exit;
    }

    /* Get the pointer to the Profile: */
    ptrMMWaveProfile = (MMWave_Profile*)profileHandle;

    /* Sanity Check: Profiles are always linked to the control module */
    DebugP_assert (ptrMMWaveProfile->ptrMMWaveMCB != NULL);

    /* Sanity Check: These API are available only in FULL configuration mode */
    DebugP_assert (ptrMMWaveProfile->ptrMMWaveMCB->initCfg.cfgMode == MMWave_ConfigurationMode_FULL);

    /* Copy over the configuration: */
    memcpy ((void*)ptrProfileCfg, (void*)&ptrMMWaveProfile->profileCfg, sizeof(rlProfileCfg_t));

    /* Setup the return value: */
    retVal = 0;

exit:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      This is an *internal* function which is used to delete the advance chirp parameter
 *      The function does not hold the critical section.
 *
 *  @param[in]  ptrMMWaveMCB
 *      Pointer to the mmWave control module
 *  @param[in]  ptrMMWaveAdvChirp
 *      Pointer to the advance chirp parameter to be deleted
 *
 *  \ingroup MMWAVE_CTRL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Not applicable
 */
static void MMWave_internalDelAdvChirp(MMWave_MCB* ptrMMWaveMCB, MMWave_AdvChirp* ptrMMWaveAdvChirp)
{
    /* Remove the advance chirp parameter from the advace chirp List  */
    MMWave_listRemoveNode ((MMWave_ListNode**)&ptrMMWaveMCB->ptrAdvChirpList, (MMWave_ListNode*)ptrMMWaveAdvChirp);

    /* Cleanup the advance chirp parameter memory: */
    HeapP_free(&gMmwHeapObj, (void *)ptrMMWaveAdvChirp);
    return;
}

/**
 *  @b Description
 *  @n
 *      This is an *internal* function which is used to delete the LUT data for an advance chirp parameter
 *      The function does not hold the critical section.
 *
 *  @param[in]  ptrMMWaveLUTData
 *      Pointer to the advance chirp parameter to be deleted
 *
 *  \ingroup MMWAVE_CTRL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Not applicable
 */
static void MMWave_internalDelAdvChirpLUTData(rlInt32_t* ptrMMWaveLUTData)
{
    /* Cleanup the advance chirp LUT data memory: */
    HeapP_free(&gMmwHeapObj, (void *)ptrMMWaveLUTData);
    return;
}

/**
 *  @b Description
 *  @n
 *      The function is used to delete the advance chirp. This will also delete all
 *      the LUT data which are still attached to the advance chirp parameter.
 *
 *  @param[in]  mmWaveHandle
 *      Handle to the mmWave module
 *  @param[in]  advChirpHandle
 *      Handle to the advance chirp parameter to be deleted
 *  @param[out] errCode
 *      Encoded Error code populated by the API on an error
 *
 *  \ingroup MMWAVE_CTRL_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MMWave_delAdvChirp
(
    MMWave_Handle           mmWaveHandle,
    MMWave_AdvChirpHandle   advChirpHandle,
    int32_t*                errCode
)
{
    MMWave_MCB*             ptrMMWaveMCB;
    MMWave_AdvChirp*        ptrMMWaveAdvChirp;
    rlInt32_t*              ptrMMWaveLUTData;
    int32_t                 retVal = MINUS_ONE;

    /* Initialize the error code: */
    *errCode = 0;

    /* Sanity Check: Validate the arguments */
    if ((mmWaveHandle == NULL) || (advChirpHandle == NULL))
    {
        /* Error: Invalid arguments */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
        goto exit;
    }

    /* Setup the pointers: */
    ptrMMWaveMCB     = (MMWave_MCB*)mmWaveHandle;
    ptrMMWaveAdvChirp = (MMWave_AdvChirp*)advChirpHandle;

    /* Sanity Check: These API are available only in FULL configuration mode */
    DebugP_assert (ptrMMWaveMCB->initCfg.cfgMode == MMWave_ConfigurationMode_FULL);

    /* Critical Section Enter: Protect the 'Advance chirp Parameter List & LUT Data' */
    SemaphoreP_pend (&(ptrMMWaveMCB->cfgSemHandle), SystemP_WAIT_FOREVER);

    /* Get the LUT Data Pointer: */
    ptrMMWaveLUTData = (rlInt32_t*) (ptrMMWaveAdvChirp->ptrLUTData);

    /* Delete the LUT Data: Use the internal API since the semaphore is already held. */
    MMWave_internalDelAdvChirpLUTData (ptrMMWaveLUTData);

    /* Delete the advanced chirp parameter: */
    MMWave_internalDelAdvChirp (ptrMMWaveMCB, ptrMMWaveAdvChirp);

    /* Critical Section Exit: */
    SemaphoreP_post (&(ptrMMWaveMCB->cfgSemHandle));

    /* Setup the return value: */
    retVal = 0;

exit:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to add the advanced chirp configuration.
 *
 *  @param[in]  mmWaveHandle
 *      Handle to the profile to which the chirp is to be added
 *  @param[in]  ptrAdvChirpCfg
 *      Pointer to the advanced chirp param configuration
 *  @param[out] errCode
 *      Encoded Error code populated by the API on an error
 *
 *  \ingroup MMWAVE_CTRL_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   Handle to the advanced chirp
 *  @retval
 *      Error   -   NULL
 */
MMWave_AdvChirpHandle MMWave_addAdvChirpParam
(
    MMWave_Handle           mmWaveHandle,
    const rlAdvChirpCfg_t*  ptrAdvChirpCfg,
    int32_t*                errCode
)
{
    MMWave_MCB*             ptrMMWaveMCB;
    MMWave_AdvChirp*        ptrMMWaveAdvChirp;
    MMWave_AdvChirpHandle   retHandle = NULL;

    /* Initialize the error code: */
    *errCode = 0;

    /* Sanity Check: Validate the arguments */
    if ((mmWaveHandle == NULL) || (ptrAdvChirpCfg == NULL))
    {
        /* Error: Invalid argument */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
        goto exit;
    }

    /* Get the pointer to the mmWave MCB: */
    ptrMMWaveMCB = (MMWave_MCB*)mmWaveHandle;

    /* Sanity Check: These API are available only in FULL configuration mode */
    DebugP_assert (ptrMMWaveMCB->initCfg.cfgMode == MMWave_ConfigurationMode_FULL);

    /* Allocate memory for the advanced chirp: */
    ptrMMWaveAdvChirp = (MMWave_AdvChirp*) HeapP_alloc(&gMmwHeapObj, (sizeof(MMWave_AdvChirp)));

    if (ptrMMWaveAdvChirp == NULL)
    {
        /* Error: Out of memory */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_ENOMEM, 0);
        goto exit;
    }

    /* Initialize the allocated memory for the advanced chirp: */
    memset ((void*)ptrMMWaveAdvChirp, 0, sizeof(MMWave_AdvChirp));

    /* Populate the advance chirp: */
    memcpy ((void *)&ptrMMWaveAdvChirp->advChirpCfg, (const void*)ptrAdvChirpCfg, sizeof(rlAdvChirpCfg_t));

    ptrMMWaveAdvChirp->ptrMMWaveMCB = ptrMMWaveMCB;

    /* Critical Section Enter: Protect the 'Advanced Chirp List' */
    SemaphoreP_pend (&(ptrMMWaveMCB->cfgSemHandle), SystemP_WAIT_FOREVER);

    /* Add the advChirp cfg to the advChirp Cfg list */
    MMWave_listCat ((MMWave_ListNode**)&ptrMMWaveMCB->ptrAdvChirpList, (MMWave_ListNode**)&ptrMMWaveAdvChirp);

    /* Critical Section Exit: */
    SemaphoreP_post (&(ptrMMWaveMCB->cfgSemHandle));

    /* Setup the return handle: */
    retHandle = (MMWave_AdvChirpHandle)ptrMMWaveAdvChirp;

exit:
    return retHandle;
}

/**
 *  @b Description
 *  @n
 *      The function is used to add the lut data to an existing advance chirp configuration.
 *
 *  @param[in]  advChirpHandle
 *      Handle to the advance chirp to which the lut data is to be added
 *  @param[in]  ptrLutData
 *      Pointer to the LUT data
 *  @param[out] errCode
 *      Encoded Error code populated by the API on an error
 *
 *  \ingroup MMWAVE_CTRL_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   Handle to the LUT data
 *  @retval
 *      Error   -   NULL
 */
MMWave_LUTDataHandle MMWave_addAdvChirpLUTData
(
    MMWave_AdvChirpHandle    advChirpHandle,
    const rlInt32_t*         ptrLutData,
    int32_t*                 errCode
)
{
    MMWave_AdvChirp*        ptrMMWaveAdvChirp;
    rlInt32_t*              ptrMMWaveLUTData;
    rlUInt16_t              numOfPatterns;
    MMWave_LUTDataHandle    retHandle = NULL;

    /* Initialize the error code: */
    *errCode = 0;

    /* Sanity Check: Validate the arguments */
    if ((advChirpHandle == NULL) || (ptrLutData == NULL))
    {
        /* Error: Invalid argument */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
        goto exit;
    }

    /* Get the advance chirp handle: */
    ptrMMWaveAdvChirp = (MMWave_AdvChirp*)advChirpHandle;

    /* Sanity Check: Each chirp parameter cfg is linked to the mmWave module */
    DebugP_assert (ptrMMWaveAdvChirp->ptrMMWaveMCB != NULL);

    /* Sanity Check: These API are available only in FULL configuration mode */
    DebugP_assert (ptrMMWaveAdvChirp->ptrMMWaveMCB->initCfg.cfgMode == MMWave_ConfigurationMode_FULL);

    numOfPatterns = ptrMMWaveAdvChirp->advChirpCfg.numOfPatterns;

    /* Allocate memory for the lut data: */
    ptrMMWaveLUTData = (rlInt32_t*) HeapP_alloc(&gMmwHeapObj, sizeof(rlInt32_t)*numOfPatterns);

    if (ptrMMWaveLUTData == NULL)
    {
        /* Error: Out of memory */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_ENOMEM, 0);
        goto exit;
    }

    /* Initialize the allocated memory for the lut data: */
    memset ((void*)ptrMMWaveLUTData, 0, sizeof(rlInt32_t)*numOfPatterns);

    /* Populate the lut data: */
    memcpy ((void *)ptrMMWaveLUTData, (const void*)ptrLutData, sizeof(rlInt32_t)*numOfPatterns);

    /* Link the LUT data buffer to the advance chirp parameter handle*/
    ptrMMWaveAdvChirp->ptrLUTData = ptrMMWaveLUTData;

    /* Setup the return handle: */
    retHandle = (MMWave_LUTDataHandle)ptrMMWaveLUTData;

exit:
    return retHandle;
}

/**
 *  @b Description
 *  @n
 *      The function is used to get the number of advanced chirp configuration parameters
 *      that have been configured.
 *
 *  @param[in]  mmWaveHandle
 *      Handle to the mmWave module
 *  @param[out] numAdvChirpCfgParams
 *      Number of added advanced chirp configuration parameters populated by the API
 *  @param[out] errCode
 *      Encoded Error code populated by the API on an error
 *
 *  \ingroup MMWAVE_CTRL_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MMWave_getNumAdvChirpParams
(
    MMWave_Handle mmWaveHandle,
    uint32_t* numAdvChirpCfgParams,
    int32_t* errCode
)
{
    MMWave_MCB*           ptrMMWaveMCB;
    MMWave_AdvChirp*      ptrMMWaveAdvChirp;
    int32_t               retVal = MINUS_ONE;

    /* Initialize the error code: */
    *errCode = 0;

    /* Sanity Check: Validate the arguments */
    if ((mmWaveHandle == NULL) || (numAdvChirpCfgParams == NULL))
    {
        /* Error: Invalid arguments */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
        goto exit;
    }

    /* Get the pointer to the mmWave control module: */
    ptrMMWaveMCB = (MMWave_MCB*)mmWaveHandle;

    /* Sanity Check: These API are available only in FULL configuration mode */
    DebugP_assert (ptrMMWaveMCB->initCfg.cfgMode == MMWave_ConfigurationMode_FULL);

    /* Critical Section Enter: Protect the 'advChirpCfgParam List' */
    SemaphoreP_pend (&(ptrMMWaveMCB->cfgSemHandle), SystemP_WAIT_FOREVER);

    /* Initialize the number of advanced chirp params */
    *numAdvChirpCfgParams = 0U;

    /* Cycle through the advChirpParam cfg list */
    ptrMMWaveAdvChirp = (MMWave_AdvChirp*)MMWave_listGetHead ((MMWave_ListNode**)&ptrMMWaveMCB->ptrAdvChirpList);
    while (ptrMMWaveAdvChirp != NULL)
    {
        /* Increment the number of advChirp Params */
        *numAdvChirpCfgParams = *numAdvChirpCfgParams + 1U;
        ptrMMWaveAdvChirp = (MMWave_AdvChirp*)MMWave_listGetNext ((MMWave_ListNode*)ptrMMWaveAdvChirp);
    }

    /* Critical Section Exit: */
    SemaphoreP_post (&(ptrMMWaveMCB->cfgSemHandle));

    /* Setup the return value: */
    retVal = 0;

exit:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function can be used by the application to get the advChirp cfg handle
 *      at the specified index. If the index exceeds the configured number of advChirp params,
 *      the function will fail with the error code.
 *
 *  @param[in]  mmWaveHandle
 *      Handle to the mmWave module
 *  @param[in]  advChirpParamIndex
 *      chirp param index for which the handle is needed. Set to 1 to get the
 *      first param index, etc
 *  @param[out] advChirpHandle
 *      Populated advanced chirp Cfg handle
 *  @param[out] errCode
 *      Encoded Error code populated by the API on an error
 *
 *  \ingroup MMWAVE_CTRL_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MMWave_getAdvChirpHandle
(
    MMWave_Handle             mmWaveHandle,
    uint32_t                  advChirpParamIndex,
    MMWave_AdvChirpHandle*    advChirpHandle,
    int32_t*                  errCode
)
{
    MMWave_MCB*         ptrMMWaveMCB;
    MMWave_AdvChirp*    ptrMMWaveAdvChirp;
    int32_t             retVal = MINUS_ONE;

    /* Initialize the error code: */
    *errCode     = 0;

    /* Sanity Check: Validate the arguments */
    if ((mmWaveHandle == NULL) || (advChirpHandle == NULL))
    {
        /* Error: Invalid arguments */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
        goto exit;
    }

    /* Get the pointer to the mmWave module: */
    ptrMMWaveMCB = (MMWave_MCB*)mmWaveHandle;

    /* Sanity Check: These API are available only in FULL configuration mode */
    DebugP_assert (ptrMMWaveMCB->initCfg.cfgMode == MMWave_ConfigurationMode_FULL);

    /* Initialize the handle */
    *advChirpHandle = NULL;

    /* Critical Section Enter: Protect the 'advance chirp cfg List' */
    SemaphoreP_pend (&(ptrMMWaveMCB->cfgSemHandle), SystemP_WAIT_FOREVER);

    /* Get the head of the list: */
    ptrMMWaveAdvChirp = (MMWave_AdvChirp*)MMWave_listGetHead ((MMWave_ListNode**)&ptrMMWaveMCB->ptrAdvChirpList);
    while (ptrMMWaveAdvChirp != NULL)
    {
        /* Is this what we are looking for? */
        if(advChirpParamIndex == ptrMMWaveAdvChirp->advChirpCfg.chirpParamIdx)
        {
            /* YES: Setup the advance chirp handle */
            *advChirpHandle = (MMWave_AdvChirpHandle)ptrMMWaveAdvChirp;
            retVal = 0;
            break;
        }

        /* Get the next element: */
        ptrMMWaveAdvChirp = (MMWave_AdvChirp*)MMWave_listGetNext ((MMWave_ListNode*)ptrMMWaveAdvChirp);
    }

    /* Critical Section Exit: */
    SemaphoreP_post (&(ptrMMWaveMCB->cfgSemHandle));

    /* Did we find a match? */
    if (ptrMMWaveAdvChirp == NULL)
    {
        /* Error: No Matching advanced chirp parameter identifier not found. Setup the error code. */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_ENOTFOUND, 0);
    }

exit:
    return retVal;
}


/**
 *  @b Description
 *  @n
 *      The function is used to get the advanced chirp configuration parameters
 *      that have been configured.
 *
 *  @param[in]  advChirpHandle
 *      Handle to the advance chirp parameter
 *  @param[out] ptrAdvChirpCfg
 *      Pointer to the advanced chirp configuration parameters populated by the API
 *  @param[out] errCode
 *      Encoded Error code populated by the API on an error
 *
 *  \ingroup MMWAVE_CTRL_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MMWave_getAdvChirpParams
(
    MMWave_AdvChirpHandle   advChirpHandle,
    rlAdvChirpCfg_t*        ptrAdvChirpCfg,
    int32_t*                errCode
)
{
    MMWave_AdvChirp*      ptrMMWaveAdvChirp;
    int32_t               retVal = MINUS_ONE;

    /* Initialize the error code: */
    *errCode = 0;

    /* Sanity Check: Validate the arguments */
    if ((advChirpHandle == NULL) || (ptrAdvChirpCfg == NULL))
    {
        /* Error: Invalid arguments */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
        goto exit;
    }

    /* Get the pointer to the advance chirp: */
    ptrMMWaveAdvChirp = (MMWave_AdvChirp*)advChirpHandle;

    /* Sanity Check: These API are available only in FULL configuration mode */
    DebugP_assert (ptrMMWaveAdvChirp->ptrMMWaveMCB->initCfg.cfgMode == MMWave_ConfigurationMode_FULL);

    /* copy over the configuration */
    memcpy((void*)ptrAdvChirpCfg, (void*)&ptrMMWaveAdvChirp->advChirpCfg, sizeof(rlAdvChirpCfg_t));

    /* Setup the return value: */
    retVal = 0;

exit:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to get the pointer to LUT data buffer of each chirp parameter
 *
 *  @param[in]  advChirpHandle
 *      Handle to the advance chirp parameter
 *  @param[out] ptrLUTData
 *      Pointer to the LUT data buffer populated by the API
 *  @param[out] errCode
 *      Encoded Error code populated by the API on an error
 *
 *  \ingroup MMWAVE_CTRL_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MMWave_getLUTData
(
    MMWave_AdvChirpHandle   advChirpHandle,
    rlInt32_t**             ptrLUTData,
    int32_t*                errCode
)
{
    MMWave_AdvChirp*      ptrMMWaveAdvChirp;
    int32_t               retVal = MINUS_ONE;

    /* Initialize the error code: */
    *errCode = 0;

    /* Sanity Check: Validate the arguments */
    if ((advChirpHandle == NULL) || (ptrLUTData==NULL))
    {
        /* Error: Invalid arguments */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
        goto exit;
    }

    /* Get the pointer to the advance chirp: */
    ptrMMWaveAdvChirp = (MMWave_AdvChirp*)advChirpHandle;

    /* Sanity Check: These API are available only in FULL configuration mode */
    DebugP_assert (ptrMMWaveAdvChirp->ptrMMWaveMCB->initCfg.cfgMode == MMWave_ConfigurationMode_FULL);

    /* cpoy the address of lut Data buffer */
    *ptrLUTData = ptrMMWaveAdvChirp->ptrLUTData;

    /* Setup the return value: */
    if(*ptrLUTData == NULL)
    {
        /* Error: No Matching LUT Data Buffer identifier not found. Setup the error code. */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_ENOTFOUND, 0);
        goto exit;
    }
    retVal = 0;

exit:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to add BPM configuration
 *      to one or more chirps.
 *
 *  @param[in]  mmWaveHandle
 *      Handle to the mmWave module
 *  @param[in]  ptrBpmChirp
 *      Pointer to the BPM configuration
 *  @param[out] errCode
 *      Encoded Error code populated by the API on an error
 *
 *  \ingroup MMWAVE_CTRL_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   Handle to the BPM configuration
 *  @retval
 *      Error   -   NULL
 */
MMWave_BpmChirpHandle MMWave_addBpmChirp
(
    MMWave_Handle           mmWaveHandle,
    const rlBpmChirpCfg_t*  ptrBpmChirp,
    int32_t*                errCode
)
{
    MMWave_MCB*               ptrMMWaveMCB;
    MMWave_BpmChirp*          ptrMMWaveBpmChirp;
    MMWave_BpmChirpHandle     retHandle = NULL;

    /* Initialize the error code: */
    *errCode = 0;

    /* Sanity Check: Validate the arguments */
    if ((mmWaveHandle == NULL) || (ptrBpmChirp == NULL))
    {
        /* Error: Invalid arguments */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
        goto exit;
    }

    /* Get the pointer to the mmWave MCB: */
    ptrMMWaveMCB = (MMWave_MCB*)mmWaveHandle;

    /* Sanity Check: These API are available only in FULL configuration mode */
    DebugP_assert (ptrMMWaveMCB->initCfg.cfgMode == MMWave_ConfigurationMode_FULL);

    /* Allocate memory for the BPM configuration: */
    ptrMMWaveBpmChirp = HeapP_alloc(&gMmwHeapObj, (sizeof(MMWave_BpmChirp)));
    if (ptrMMWaveBpmChirp == NULL)
    {
        /* Error: Out of memory */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_ENOMEM, 0);
        goto exit;
    }

    /* Initialize the allocated memory: */
    memset ((void *)ptrMMWaveBpmChirp, 0, sizeof(MMWave_BpmChirp));

    /* Populate the BPM cfg: */
    memcpy ((void*)&ptrMMWaveBpmChirp->bpmChirp, (const void*)ptrBpmChirp, sizeof(rlBpmChirpCfg_t));
    ptrMMWaveBpmChirp->ptrMMWaveMCB = ptrMMWaveMCB;

    /* Critical Section Enter: Protect the 'BPM cfg List' */
    SemaphoreP_pend (&(ptrMMWaveMCB->cfgSemHandle), SystemP_WAIT_FOREVER);

    /* Add the BPM cfg to the BPM cfg List  */
    MMWave_listCat ((MMWave_ListNode**)&ptrMMWaveMCB->ptrBpmChirpList, (MMWave_ListNode**)&ptrMMWaveBpmChirp);

    /* Critical Section Exit: */
    SemaphoreP_post (&(ptrMMWaveMCB->cfgSemHandle));

    /* BPM configuration has been successfully registered */
    retHandle = (MMWave_BpmChirpHandle)ptrMMWaveBpmChirp;

exit:
    return retHandle;
}

/**
 *  @b Description
 *  @n
 *      This is an *internal* function which is used to delete the BPM cfg
 *      The function does not hold the critical section.
 *
 *  @param[in]  ptrMMWaveMCB
 *      Pointer to the mmWave control module
 *  @param[in]  ptrMMWaveBpmChirp
 *      Pointer to the BPM cfg to be deleted
 *
 *  \ingroup MMWAVE_CTRL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Not applicable
 */
static void MMWave_internalDelBpmChirp (MMWave_MCB* ptrMMWaveMCB, MMWave_BpmChirp* ptrMMWaveBpmChirp)
{
    /* Remove the BPM cfg from the BPM cfg List  */
    MMWave_listRemoveNode ((MMWave_ListNode**)&ptrMMWaveMCB->ptrBpmChirpList, (MMWave_ListNode*)ptrMMWaveBpmChirp);

    /* Cleanup the BPM cfg memory: */
    HeapP_free(&gMmwHeapObj, (void *)ptrMMWaveBpmChirp);
    return;
}

/**
 *  @b Description
 *  @n
 *      The function is used to delete the BPM cfg.
 *
 *  @param[in]  mmWaveHandle
 *      Handle to the mmWave module
 *  @param[in]  bpmChirpHandle
 *      Handle to the BPM cfg to be deleted
 *  @param[out] errCode
 *      Encoded Error code populated by the API on an error
 *
 *  \ingroup MMWAVE_CTRL_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MMWave_delBpmChirp
(
    MMWave_Handle           mmWaveHandle,
    MMWave_BpmChirpHandle   bpmChirpHandle,
    int32_t*                errCode
)
{
    MMWave_MCB*             ptrMMWaveMCB;
    MMWave_BpmChirp*        ptrMMWaveBpmChirp;
    int32_t                 retVal = MINUS_ONE;

    /* Initialize the error code: */
    *errCode = 0;

    /* Sanity Check: Validate the arguments */
    if ((mmWaveHandle == NULL) || (bpmChirpHandle == NULL))
    {
        /* Error: Invalid arguments */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
        goto exit;
    }

    /* Setup the pointers: */
    ptrMMWaveMCB     = (MMWave_MCB*)mmWaveHandle;
    ptrMMWaveBpmChirp  = (MMWave_BpmChirp*)bpmChirpHandle;

    /* Sanity Check: These API are available only in FULL configuration mode */
    DebugP_assert (ptrMMWaveMCB->initCfg.cfgMode == MMWave_ConfigurationMode_FULL);

    /* Critical Section Enter: Protect the 'BPM configuration List' */
    SemaphoreP_pend (&(ptrMMWaveMCB->cfgSemHandle), SystemP_WAIT_FOREVER);

    /* Delete the BPM cfg: */
    MMWave_internalDelBpmChirp (ptrMMWaveMCB, ptrMMWaveBpmChirp);

    /* Critical Section Exit: */
    SemaphoreP_post (&(ptrMMWaveMCB->cfgSemHandle));

    /* Setup the return value: */
    retVal = 0;

exit:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to get the number of BPM configurations
 *      that have been added.
 *
 *  @param[in]  mmWaveHandle
 *      Handle to the mmWave module
 *  @param[out] numBpmChirp
 *      Number of added BPM configurations populated by the API
 *  @param[out] errCode
 *      Encoded Error code populated by the API on an error
 *
 *  \ingroup MMWAVE_CTRL_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MMWave_getNumBpmChirp(MMWave_Handle mmWaveHandle, uint32_t* numBpmChirp, int32_t* errCode)
{
    MMWave_MCB*           ptrMMWaveMCB;
    MMWave_BpmChirp*      ptrMMWaveBpmChirp;
    int32_t               retVal = MINUS_ONE;

    /* Initialize the error code: */
    *errCode = 0;

    /* Sanity Check: Validate the arguments */
    if ((mmWaveHandle == NULL) || (numBpmChirp == NULL))
    {
        /* Error: Invalid arguments */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
        goto exit;
    }

    /* Get the pointer to the mmWave control module: */
    ptrMMWaveMCB = (MMWave_MCB*)mmWaveHandle;

    /* Sanity Check: These API are available only in FULL configuration mode */
    DebugP_assert (ptrMMWaveMCB->initCfg.cfgMode == MMWave_ConfigurationMode_FULL);

    /* Critical Section Enter: Protect the 'BPM cfg List' */
    SemaphoreP_pend (&(ptrMMWaveMCB->cfgSemHandle), SystemP_WAIT_FOREVER);

    /* Initialize the number of BPM cfgs */
    *numBpmChirp = 0U;

    /* Cycle through the BPM cfg list */
    ptrMMWaveBpmChirp = (MMWave_BpmChirp*)MMWave_listGetHead ((MMWave_ListNode**)&ptrMMWaveMCB->ptrBpmChirpList);
    while (ptrMMWaveBpmChirp != NULL)
    {
        /* Increment the number of BPM cfgs */
        *numBpmChirp = *numBpmChirp + 1U;
        ptrMMWaveBpmChirp = (MMWave_BpmChirp*)MMWave_listGetNext ((MMWave_ListNode*)ptrMMWaveBpmChirp);
    }

    /* Critical Section Exit: */
    SemaphoreP_post (&(ptrMMWaveMCB->cfgSemHandle));

    /* Setup the return value: */
    retVal = 0;

exit:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function can be used by the application to get the BPM cfg handle
 *      at the specified index. If the index exceeds the number of BPM configurations
 *      the function will fail with the error code.
 *
 *  @param[in]  mmWaveHandle
 *      Handle to the mmWave module
 *  @param[in]  bpmChirpIndex
 *      BPM cfg index for which the handle is needed. Set to 1 to get the
 *      first BPM configuration index, etc
 *  @param[out] bpmChirpHandle
 *      Populated BPM Cfg handle
 *  @param[out] errCode
 *      Encoded Error code populated by the API on an error
 *
 *  \ingroup MMWAVE_CTRL_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MMWave_getBpmChirpHandle
(
    MMWave_Handle             mmWaveHandle,
    uint32_t                  bpmChirpIndex,
    MMWave_BpmChirpHandle*    bpmChirpHandle,
    int32_t*                  errCode
)
{
    MMWave_MCB*         ptrMMWaveMCB;
    MMWave_BpmChirp*    ptrMMWaveBpmChirp;
    uint32_t            index  = 1U;
    int32_t             retVal = MINUS_ONE;
    int32_t             endProcessing = 0;

    /* Initialize the error code: */
    *errCode     = 0;

    /* Sanity Check: Validate the arguments */
    if ((mmWaveHandle == NULL) || (bpmChirpHandle == NULL))
    {
        /* Error: Invalid arguments */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
        goto exit;
    }

    /* Get the pointer to the mmWave control module: */
    ptrMMWaveMCB = (MMWave_MCB*)mmWaveHandle;

    /* Sanity Check: These API are available only in FULL configuration mode */
    DebugP_assert (ptrMMWaveMCB->initCfg.cfgMode == MMWave_ConfigurationMode_FULL);

    /* Initialize the handle */
    *bpmChirpHandle = NULL;

    /* Critical Section Enter: Protect the 'BPM cfg List' */
    SemaphoreP_pend (&(ptrMMWaveMCB->cfgSemHandle), SystemP_WAIT_FOREVER);

    /* Get the head of the list: */
    ptrMMWaveBpmChirp = (MMWave_BpmChirp*)MMWave_listGetHead ((MMWave_ListNode**)&ptrMMWaveMCB->ptrBpmChirpList);
    while (endProcessing == 0)
    {
        /* Have we reached the end of the list? */
        if (ptrMMWaveBpmChirp == NULL)
        {
            /* YES: Control comes here indicates that the index specified exceeds the
             * number of BPM cfgs. We are done with the processing */
            *errCode      = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
            endProcessing = 1;
        }
        else
        {
            /* Is this what we are looking for? */
            if (index == bpmChirpIndex)
            {
                /* YES: Setup the handle. */
                *bpmChirpHandle  = (MMWave_BpmChirpHandle)ptrMMWaveBpmChirp;
                retVal        = 0;
                endProcessing = 1;
            }

            /* Get the next element: */
            index = index + 1U;
            ptrMMWaveBpmChirp = (MMWave_BpmChirp*)MMWave_listGetNext ((MMWave_ListNode*)ptrMMWaveBpmChirp);
        }
    }

    /* Critical Section Exit: */
    SemaphoreP_post (&(ptrMMWaveMCB->cfgSemHandle));

exit:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to get the BPM configuration given
 *      the BPM cfg handle
 *
 *  @param[in]  bpmChirpHandle
 *      Handle to the BPM configuration
 *  @param[out] ptrBpmChirp
 *      Pointer to the BPM configuration populated by the API
 *  @param[out] errCode
 *      Encoded Error code populated by the API on an error
 *
 *  \ingroup MMWAVE_CTRL_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MMWave_getBpmChirpCfg
(
    MMWave_BpmChirpHandle    bpmChirpHandle,
    rlBpmChirpCfg_t*         ptrBpmChirp,
    int32_t*                 errCode
)
{
    MMWave_BpmChirp*     ptrMMWaveBpmChirp;
    int32_t              retVal = MINUS_ONE;

    /* Initialize the error code: */
    *errCode = 0;

    /* Sanity Check: Validate the arguments */
    if ((bpmChirpHandle == NULL) || (ptrBpmChirp == NULL))
    {
        /* Error: Invalid arguments */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
        goto exit;
    }

    /* Get the pointer to the BPM configuration: */
    ptrMMWaveBpmChirp = (MMWave_BpmChirp*)bpmChirpHandle;

    /* Sanity Check */
    DebugP_assert (ptrMMWaveBpmChirp->ptrMMWaveMCB != NULL);

    /* Sanity Check: These API are available only in FULL configuration mode */
    DebugP_assert (ptrMMWaveBpmChirp->ptrMMWaveMCB->initCfg.cfgMode == MMWave_ConfigurationMode_FULL);

    /* Copy over the configuration: */
    memcpy ((void*)ptrBpmChirp, (void*)&ptrMMWaveBpmChirp->bpmChirp, sizeof(rlBpmChirpCfg_t));

    /* Setup the return value: */
    retVal = 0;

exit:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to add chirp level phase shift
 *      configuration to one or more chirps.
 *
 *  @param[in]  mmWaveHandle
 *      Handle to the mmWave module
 *  @param[in]  ptrPhaseShiftChirp
 *      Pointer to the chirp level phase shifter configuration
 *  @param[out] errCode
 *      Encoded Error code populated by the API on an error
 *
 *  \ingroup MMWAVE_CTRL_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   Handle to the chirp level phase shifter configuration
 *  @retval
 *      Error   -   NULL
 */
MMWave_PhaseShiftChirpHandle MMWave_addPhaseShiftChirp
(
        MMWave_Handle               mmWaveHandle,
        const rlRfPhaseShiftCfg_t*  ptrPhaseShiftChirp,
        int32_t*                    errCode
    )
{
    MMWave_MCB*                      ptrMMWaveMCB;
    MMWave_phaseShiftChirp*          ptrMMWavePhaseShiftChirp;
    MMWave_PhaseShiftChirpHandle     retHandle = NULL;

    /* Initialize the error code: */
    *errCode = 0;

    /* Sanity Check: Validate the arguments */
    if ((mmWaveHandle == NULL) || (ptrPhaseShiftChirp == NULL))
    {
        /* Error: Invalid arguments */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
        goto exit;
    }

    /* Get the pointer to the mmWave MCB: */
    ptrMMWaveMCB = (MMWave_MCB*)mmWaveHandle;

    /* Sanity Check: These API are available only in FULL configuration mode */
    DebugP_assert (ptrMMWaveMCB->initCfg.cfgMode == MMWave_ConfigurationMode_FULL);

    /* Allocate memory for the chirp level phase shifter configuration: */
    ptrMMWavePhaseShiftChirp = HeapP_alloc(&gMmwHeapObj, (sizeof(MMWave_phaseShiftChirp)));
    if (ptrMMWavePhaseShiftChirp == NULL)
    {
        /* Error: Out of memory */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_ENOMEM, 0);
        goto exit;
    }

    /* Initialize the allocated memory: */
    memset ((void *)ptrMMWavePhaseShiftChirp, 0, sizeof(MMWave_phaseShiftChirp));

    /* Populate the chirp level phase shifter cfg: */
    memcpy ((void*)&ptrMMWavePhaseShiftChirp->phaseShiftChirp, (const void*)ptrPhaseShiftChirp, sizeof(rlRfPhaseShiftCfg_t));
    ptrMMWavePhaseShiftChirp->ptrMMWaveMCB = ptrMMWaveMCB;

    /* Critical Section Enter: Protect the 'chirp level phase shifter cfg List' */
    SemaphoreP_pend (&ptrMMWaveMCB->cfgSemHandle, SystemP_WAIT_FOREVER);

    /* Add the chirp level phase shifter cfg to the List  */
    MMWave_listCat ((MMWave_ListNode**)&ptrMMWaveMCB->ptrPhaseShiftChirpList, (MMWave_ListNode**)&ptrMMWavePhaseShiftChirp);

    /* Critical Section Exit: */
    SemaphoreP_post (&ptrMMWaveMCB->cfgSemHandle);

    /* chirp level phase shifter configuration has been successfully registered */
    retHandle = (MMWave_PhaseShiftChirpHandle)ptrMMWavePhaseShiftChirp;

exit:
    return retHandle;
}

/**
 *  @b Description
 *  @n
 *      This is an *internal* function which is used to delete the chirp level phase shifter cfg
 *      The function does not hold the critical section.
 *
 *  @param[in]  ptrMMWaveMCB
 *      Pointer to the mmWave control module
 *  @param[in]  ptrMMWavePhaseShiftChirp
 *      Pointer to the chirp level phase shifter cfg to be deleted
 *
 *  \ingroup MMWAVE_CTRL_INTERNAL_FUNCTION
 *
 *  @retval
 *      Not applicable
 */
static void MMWave_internalDelPhaseShiftChirp (MMWave_MCB* ptrMMWaveMCB, MMWave_phaseShiftChirp* ptrMMWavePhaseShiftChirp)
{
    /* Remove the chirp level phase shifter cfg from the List  */
    MMWave_listRemoveNode ((MMWave_ListNode**)&ptrMMWaveMCB->ptrPhaseShiftChirpList, (MMWave_ListNode*)ptrMMWavePhaseShiftChirp);

    /* Cleanup the chirp level phase shifter cfg memory: */
    HeapP_free(&gMmwHeapObj, (void *)ptrMMWavePhaseShiftChirp);
    return;
}

/**
 *  @b Description
 *  @n
 *      The function is used to delete the chirp level phase shifter cfg.
 *
 *  @param[in]  mmWaveHandle
 *      Handle to the mmWave module
 *  @param[in]  phaseShiftChirpHandle
 *      Handle to the chirp level phase shifter cfg to be deleted
 *  @param[out] errCode
 *      Encoded Error code populated by the API on an error
 *
 *  \ingroup MMWAVE_CTRL_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MMWave_delPhaseShiftChirp
(
    MMWave_Handle                  mmWaveHandle,
    MMWave_PhaseShiftChirpHandle   phaseShiftChirpHandle,
    int32_t*                       errCode
)
{
    MMWave_MCB*                     ptrMMWaveMCB;
    MMWave_phaseShiftChirp*         ptrMMWavePhaseShiftChirp;
    int32_t                         retVal = MINUS_ONE;

    /* Initialize the error code: */
    *errCode = 0;

    /* Sanity Check: Validate the arguments */
    if ((mmWaveHandle == NULL) || (phaseShiftChirpHandle == NULL))
    {
        /* Error: Invalid arguments */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
        goto exit;
    }

    /* Setup the pointers: */
    ptrMMWaveMCB     = (MMWave_MCB*)mmWaveHandle;
    ptrMMWavePhaseShiftChirp  = (MMWave_phaseShiftChirp*)phaseShiftChirpHandle;

    /* Sanity Check: These API are available only in FULL configuration mode */
    DebugP_assert (ptrMMWaveMCB->initCfg.cfgMode == MMWave_ConfigurationMode_FULL);

    /* Critical Section Enter: Protect the 'chirp level phase shifter configuration List' */
    SemaphoreP_pend (&ptrMMWaveMCB->cfgSemHandle, SystemP_WAIT_FOREVER);

    /* Delete the chirp level phase shifter cfg: */
    MMWave_internalDelPhaseShiftChirp (ptrMMWaveMCB, ptrMMWavePhaseShiftChirp);

    /* Critical Section Exit: */
    SemaphoreP_post (&ptrMMWaveMCB->cfgSemHandle);

    /* Setup the return value: */
    retVal = 0;

exit:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to get the number of chirp level phase shifter configurations
 *      that have been added.
 *
 *  @param[in]  mmWaveHandle
 *      Handle to the mmWave module
 *  @param[out] numPhaseShiftChirp
 *      Number of added chirp level phase shifter configurations populated by the API
 *  @param[out] errCode
 *      Encoded Error code populated by the API on an error
 *
 *  \ingroup MMWAVE_CTRL_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MMWave_getNumPhaseShiftChirp(MMWave_Handle mmWaveHandle, uint32_t* numPhaseShiftChirp, int32_t* errCode)
{
    MMWave_MCB*                     ptrMMWaveMCB;
    MMWave_phaseShiftChirp*         ptrMMWavePhaseShiftChirp;
    int32_t                         retVal = MINUS_ONE;

    /* Initialize the error code: */
    *errCode = 0;

    /* Sanity Check: Validate the arguments */
    if ((mmWaveHandle == NULL) || (numPhaseShiftChirp == NULL))
    {
        /* Error: Invalid arguments */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
        goto exit;
    }

    /* Get the pointer to the mmWave control module: */
    ptrMMWaveMCB = (MMWave_MCB*)mmWaveHandle;

    /* Sanity Check: These API are available only in FULL configuration mode */
    DebugP_assert (ptrMMWaveMCB->initCfg.cfgMode == MMWave_ConfigurationMode_FULL);

    /* Critical Section Enter: Protect the 'chirp level phase shifter cfg List' */
    SemaphoreP_pend (&ptrMMWaveMCB->cfgSemHandle, SystemP_WAIT_FOREVER);

    /* Initialize the number of chirp level phase shifter cfgs */
    *numPhaseShiftChirp = 0U;

    /* Cycle through the chirp level phase shifter cfg list */
    ptrMMWavePhaseShiftChirp = (MMWave_phaseShiftChirp*)MMWave_listGetHead ((MMWave_ListNode**)&ptrMMWaveMCB->ptrPhaseShiftChirpList);
    while (ptrMMWavePhaseShiftChirp != NULL)
    {
        /* Increment the number of chirp level phase shifter cfgs */
        *numPhaseShiftChirp = *numPhaseShiftChirp + 1U;
        ptrMMWavePhaseShiftChirp = (MMWave_phaseShiftChirp*)MMWave_listGetNext ((MMWave_ListNode*)ptrMMWavePhaseShiftChirp);
    }

    /* Critical Section Exit: */
    SemaphoreP_post (&ptrMMWaveMCB->cfgSemHandle);

    /* Setup the return value: */
    retVal = 0;

exit:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function can be used by the application to get the chirp level phase shifter cfg handle
 *      at the specified index. If the index exceeds the number of chirp level phase shifter configurations
 *      the function will fail with the error code.
 *
 *  @param[in]  mmWaveHandle
 *      Handle to the mmWave module
 *  @param[in]  phaseShiftChirpIndex
 *      chirp level phase shifter cfg index for which the handle is needed. Set to 1 to get the
 *      first chirp level phase shifter configuration index, etc
 *  @param[out] phaseShiftChirpHandle
 *      Populated chirp level phase shifter Cfg handle
 *  @param[out] errCode
 *      Encoded Error code populated by the API on an error
 *
 *  \ingroup MMWAVE_CTRL_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MMWave_getPhaseShiftChirpHandle
(
    MMWave_Handle                   mmWaveHandle,
    uint32_t                        phaseShiftChirpIndex,
    MMWave_PhaseShiftChirpHandle*   phaseShiftChirpHandle,
    int32_t*                        errCode
)
{
    MMWave_MCB*                 ptrMMWaveMCB;
    MMWave_phaseShiftChirp*     ptrMMWavePhaseShiftChirp;
    uint32_t                    index  = 1U;
    int32_t                     retVal = MINUS_ONE;
    int32_t                     endProcessing = 0;

    /* Initialize the error code: */
    *errCode     = 0;

    /* Sanity Check: Validate the arguments */
    if ((mmWaveHandle == NULL) || (phaseShiftChirpHandle == NULL))
    {
        /* Error: Invalid arguments */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
        goto exit;
    }

    /* Get the pointer to the mmWave control module: */
    ptrMMWaveMCB = (MMWave_MCB*)mmWaveHandle;

    /* Sanity Check: These API are available only in FULL configuration mode */
    DebugP_assert (ptrMMWaveMCB->initCfg.cfgMode == MMWave_ConfigurationMode_FULL);

    /* Initialize the handle */
    *phaseShiftChirpHandle = NULL;

    /* Critical Section Enter: Protect the 'chirp level phase shifter cfg List' */
    SemaphoreP_pend (&ptrMMWaveMCB->cfgSemHandle, SystemP_WAIT_FOREVER);

    /* Get the head of the list: */
    ptrMMWavePhaseShiftChirp = (MMWave_phaseShiftChirp*)MMWave_listGetHead ((MMWave_ListNode**)&ptrMMWaveMCB->ptrPhaseShiftChirpList);
    while (endProcessing == 0)
    {
        /* Have we reached the end of the list? */
        if (ptrMMWavePhaseShiftChirp == NULL)
        {
            /* YES: Control comes here indicates that the index specified exceeds the
             * number of chirp level phase shifter cfgs. We are done with the processing */
            *errCode      = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
            endProcessing = 1;
        }
        else
        {
            /* Is this what we are looking for? */
            if (index == phaseShiftChirpIndex)
            {
                /* YES: Setup the handle. */
                *phaseShiftChirpHandle  = (MMWave_PhaseShiftChirpHandle)ptrMMWavePhaseShiftChirp;
                retVal        = 0;
                endProcessing = 1;
            }

            /* Get the next element: */
            index = index + 1U;
            ptrMMWavePhaseShiftChirp = (MMWave_phaseShiftChirp*)MMWave_listGetNext ((MMWave_ListNode*)ptrMMWavePhaseShiftChirp);
        }
    }

    /* Critical Section Exit: */
    SemaphoreP_post (&ptrMMWaveMCB->cfgSemHandle);

exit:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to get the chirp level phase shifter configuration given
 *      the chirp level phase shifter cfg handle
 *
 *  @param[in]  phaseShiftChirpHandle
 *      Handle to the chirp level phase shifter configuration
 *  @param[out] ptrPhaseShiftChirp
 *      Pointer to the chirp level phase shifter configuration populated by the API
 *  @param[out] errCode
 *      Encoded Error code populated by the API on an error
 *
 *  \ingroup MMWAVE_CTRL_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MMWave_getPhaseShiftChirpCfg
(
    MMWave_PhaseShiftChirpHandle    phaseShiftChirpHandle,
    rlRfPhaseShiftCfg_t*            ptrPhaseShiftChirp,
    int32_t*                        errCode
)
{
    MMWave_phaseShiftChirp*     ptrMMWavePhaseShiftChirp;
    int32_t                     retVal = MINUS_ONE;

    /* Initialize the error code: */
    *errCode = 0;

    /* Sanity Check: Validate the arguments */
    if ((phaseShiftChirpHandle == NULL) || (ptrPhaseShiftChirp == NULL))
    {
        /* Error: Invalid arguments */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
        goto exit;
    }

    /* Get the pointer to the chirp level phase shifter configuration: */
    ptrMMWavePhaseShiftChirp = (MMWave_phaseShiftChirp*)phaseShiftChirpHandle;

    /* Sanity Check */
    DebugP_assert (ptrMMWavePhaseShiftChirp->ptrMMWaveMCB != NULL);

    /* Sanity Check: These API are available only in FULL configuration mode */
    DebugP_assert (ptrMMWavePhaseShiftChirp->ptrMMWaveMCB->initCfg.cfgMode == MMWave_ConfigurationMode_FULL);

    /* Copy over the configuration: */
    memcpy ((void*)ptrPhaseShiftChirp, (void*)&ptrMMWavePhaseShiftChirp->phaseShiftChirp, sizeof(rlRfPhaseShiftCfg_t));

    /* Setup the return value: */
    retVal = 0;

exit:
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to flush & clean up the configuration which is stored in
 *      the mmWave module. Due to memory constraints applications could use this API
 *      to reduce the amount of memory which is being used.
 *
 *  @param[in]  mmWaveHandle
 *      Handle to the mmWave module
 *  @param[out] errCode
 *      Encoded Error code populated by the API on an error
 *
 *  \ingroup MMWAVE_CTRL_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t MMWave_flushCfg
(
    MMWave_Handle   mmWaveHandle,
    int32_t*        errCode
)
{
    MMWave_MCB*       ptrMMWaveMCB;
    MMWave_Profile*   ptrMMWaveProfile;
    MMWave_Chirp*     ptrMMWaveChirp;
    MMWave_AdvChirp*  ptrMMWaveAdvChirp;
    rlInt32_t*        ptrLUTData;
    MMWave_BpmChirp*  ptrMMWaveBpmChirp;
    MMWave_phaseShiftChirp*  ptrMMWavePhaseShiftChirp;
    int32_t           retVal = MINUS_ONE;

    /* Initialize the error code: */
    *errCode = 0;

    /* Sanity Check: Validate the arguments */
    if (mmWaveHandle == NULL)
    {
        /* Error: Invalid argument */
        *errCode = MMWave_encodeError (MMWave_ErrorLevel_ERROR, MMWAVE_EINVAL, 0);
        goto exit;
    }

    /* Setup the pointers: */
    ptrMMWaveMCB = (MMWave_MCB*)mmWaveHandle;

    /* Sanity Check: These API are available only in FULL configuration mode */
    DebugP_assert (ptrMMWaveMCB->initCfg.cfgMode == MMWave_ConfigurationMode_FULL);

    /* Critical Section Enter: Protect lists */
    SemaphoreP_pend (&(ptrMMWaveMCB->cfgSemHandle), SystemP_WAIT_FOREVER);

    /* Cycle through all the profiles: */
    ptrMMWaveProfile = (MMWave_Profile*)MMWave_listGetHead ((MMWave_ListNode**)&ptrMMWaveMCB->ptrProfileList);
    while (ptrMMWaveProfile != NULL)
    {
        /* Cycle through all the registered chirps for that profile */
        ptrMMWaveChirp = (MMWave_Chirp*)MMWave_listGetHead ((MMWave_ListNode**)&ptrMMWaveProfile->ptrChirpList);
        while (ptrMMWaveChirp != NULL)
        {
            /* Delete the chirp: Use the internal API since the semaphore is already held. */
            MMWave_internalDelChirp (ptrMMWaveProfile, ptrMMWaveChirp);

            /* Cycle through the list again and get the new head. */
            ptrMMWaveChirp = (MMWave_Chirp*)MMWave_listGetHead ((MMWave_ListNode**)&ptrMMWaveProfile->ptrChirpList);
        }

        /* Delete the profile: */
        MMWave_internalDelProfile (ptrMMWaveMCB, ptrMMWaveProfile);

        /* Cycle through the next profile and get the new head. */
        ptrMMWaveProfile = (MMWave_Profile*)MMWave_listGetHead ((MMWave_ListNode**)&ptrMMWaveMCB->ptrProfileList);
    }

    /* Cycle through all BPM configurations: */
    ptrMMWaveBpmChirp = (MMWave_BpmChirp*)MMWave_listGetHead ((MMWave_ListNode**)&ptrMMWaveMCB->ptrBpmChirpList);
    while (ptrMMWaveBpmChirp != NULL)
    {
        /* Delete BPM configuration. Use the internal API since the semaphore is already held. */
        MMWave_internalDelBpmChirp (ptrMMWaveMCB, ptrMMWaveBpmChirp);

        /* Cycle through the list again and get the new head. */
        ptrMMWaveBpmChirp = (MMWave_BpmChirp*)MMWave_listGetHead ((MMWave_ListNode**)&ptrMMWaveMCB->ptrBpmChirpList);
    }

    /* Cycle through all chirp level phase shifter configurations: */
    ptrMMWavePhaseShiftChirp = (MMWave_phaseShiftChirp*)MMWave_listGetHead ((MMWave_ListNode**)&ptrMMWaveMCB->ptrPhaseShiftChirpList);
    while (ptrMMWavePhaseShiftChirp != NULL)
    {
        /* Delete chirp level phase shifter configuration. Use the internal API since the semaphore is already held. */
        MMWave_internalDelPhaseShiftChirp (ptrMMWaveMCB, ptrMMWavePhaseShiftChirp);

        /* Cycle through the list again and get the new head. */
        ptrMMWavePhaseShiftChirp = (MMWave_phaseShiftChirp*)MMWave_listGetHead ((MMWave_ListNode**)&ptrMMWaveMCB->ptrPhaseShiftChirpList);
    }

    /* Cycle through all the advanced chirp parameter configurations: */
    ptrMMWaveAdvChirp = (MMWave_AdvChirp*)MMWave_listGetHead ((MMWave_ListNode**)&ptrMMWaveMCB->ptrAdvChirpList);
    while (ptrMMWaveAdvChirp != NULL)
    {
        /* Delete the LUT data associated with that parameter */
        ptrLUTData = (rlInt32_t*) (ptrMMWaveAdvChirp->ptrLUTData);

        /* Delete the LUT data*/
        MMWave_internalDelAdvChirpLUTData (ptrLUTData);

        /* Delete the advance chirp: */
        MMWave_internalDelAdvChirp (ptrMMWaveMCB, ptrMMWaveAdvChirp);

        /* Cycle through the next advance chirp config and get the new head. */
        ptrMMWaveAdvChirp = (MMWave_AdvChirp*)MMWave_listGetHead ((MMWave_ListNode**)&ptrMMWaveMCB->ptrAdvChirpList);
    }

    /* Critical Section Exit: */
    SemaphoreP_post (&(ptrMMWaveMCB->cfgSemHandle));

    /* Setup the return value: */
    retVal = 0;

exit:
    return retVal;
}

