#ifndef AUDIOPLAYERAL_H_INCLUDED
#define AUDIOPLAYERAL_H_INCLUDED

#include <iostream>


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdint.h>
#include <inttypes.h>
#include <unistd.h>
#include <stdbool.h>
#include <iostream>
#include <fstream>
#include <iomanip>

#include <AL/al.h>
#include <AL/alc.h>
#include <AL/efx.h>
#include <AL/efx-presets.h>
#include <AL/alext.h>

#include <vector>
#include <algorithm>
#include <list>
#include <deque>
#include <condition_variable>
#include <math.h>

#include "vorbis/codec.h"
#include "vorbis/vorbisfile.h"
#include "ogg/ogg.h"

#include <fcntl.h>

#ifdef win32
#include <io.h>
#include <windows.h>
#include <share.h>
#include <io.h>
#include <mmsystem.h>
#endif // win32

#include <sys/stat.h>

#include <chrono>
#include <thread>
#include <tuple>
#include <atomic>
#include <mutex>

//
#include "abc.h"





std::mutex m;

class AudioPlayerAL
{
public:

    void Initialize(float volume, int panning);
    void Play();
    void Stop();
    void Seek(float f);
    float Position();
    void PlayTestTones(int instrument, int pitch);
    void SendABC(std::stringstream * abctext);
    void SendToneTuples(std::vector<std::vector<ToneTuple>> toneTuples);
    void ExportSamples();
    bool Finished();
    void SetInstrument(int id, int instrument);
    void SetPanning(int id, int panning);
    void SetVolume(float value);
    float GetVolume();
    void SetGlobalPanning(int panning);
    void SetMute(int id, bool value);
    size_t GetNumberOfTracks();
    int GetID(size_t track);
    int GetInstrument(size_t track);
    int GetXNumber(size_t track);
    int GetPanning(size_t track);
    int GetZPanning(size_t track);
    void PlayLoop();

    void UpdateABC(std::stringstream * abctext);

    void ExportWAV(std::string * filename);
    std::vector<float> GenerateWAVMono();
    std::vector<float> GenerateWAVMonof();
    std::vector<int32_t> GenerateWAVMonoI();
    std::vector<int32_t> GenerateWAVMonoI_11();
    std::vector<int32_t> ApplyToneDeltaMonoI(std::vector<int32_t> sampleBuf,
                                      const std::vector<ToneTuple>& added,
                                      const std::vector<ToneTuple>& deleted);

    std::vector<int32_t> GenerateWAVMonoI2();
    std::vector<int32_t> ApplyToneDeltaMonoI2(std::vector<int32_t> sampleBuf,
                                      const std::vector<ToneTuple>& added,
                                      const std::vector<ToneTuple>& deleted);

    std::vector<int32_t> ApplyToneDeltaMonoI_11(std::vector<int32_t> sampleBuf,
                                      const std::vector<ToneTuple>& added,
                                      const std::vector<ToneTuple>& deleted);
    std::vector<float> GenerateWAV();

    std::vector<std::vector<ToneTuple>> ToneData();

    std::vector<size_t> GetOriginalPartNumbers();

    ~AudioPlayerAL();
    
    std::atomic<int> audio_playing = 0;
    std::atomic<int> m_stop = 0;
    std::atomic<int> m_mute = 0; // 0 is not muted

    ABCInput * GetABC();

    // Returns total duration in samples (44.1kHz) for a given tone.
    // Handles fixed duration instruments (fadeouttype == 0) and fade-out instruments (fadeouttype != 0).
    uint64_t GetToneTotalDurationSamples(int instrument, int pitchIdx, int64_t toneDuration) const {
        if (instrument < 0 || instrument >= 23) return static_cast<uint64_t>(toneDuration);

        // Cowbells (instruments 9 & 10) use fixed sample index 36
        if (instrument == 9 || instrument == 10) pitchIdx = 36;

        uint32_t fadesamples = static_cast<uint32_t>(AudioPlayerAL::fadeouts[instrument] * 44100);

        if (fadeouttype[instrument] == 0) {
            // Fixed-duration instruments: duration is dictated by the loaded sample length
            if (pitchIdx >= 0 && pitchIdx < static_cast<int>(allsamples[instrument].size()) &&
                !allsamples[instrument][pitchIdx].empty()) {
                return allsamples[instrument][pitchIdx].size() / 2; // 16-bit mono = 2 bytes per sample
            }
            return 44100 * 2; // fallback estimate if sample data not yet loaded
        } else {
            // Variable-duration instruments: tone duration + fade-out duration
            size_t mixSamples = static_cast<size_t>(toneDuration) + fadesamples;
            if (pitchIdx >= 0 && pitchIdx < static_cast<int>(allsamples[instrument].size()) &&
                !allsamples[instrument][pitchIdx].empty()) {
                mixSamples = std::min(mixSamples, allsamples[instrument][pitchIdx].size() / 2);
            }
            return static_cast<uint64_t>(mixSamples);
        }
    }


private:

    ABCInput * myabc = NULL;

    std::thread* m_loadingThread = nullptr;
    std::atomic<bool> m_samplesLoaded{ false };
    std::mutex m_loadingMutex;
    std::condition_variable m_loadingCV;
    void LoadSamplesBackground();
    void WaitForSamples();

    std::vector< std::vector<  std::vector< uint8_t  >  > > allsamples;
        std::vector< std::vector<  std::vector< uint8_t  >  > > allsamples_11;

  //  std::vector< std::vector< std::vector<float> > > allsamples_f;

    std::vector<uint8_t> snd_load_file(const uint8_t* buffer, size_t size);

    // OpenAl Specifics
    ALCdevice *device;
	ALCcontext *context;
	ALboolean enumeration;

	ALfloat listenerOri[6] = { 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f };  // Audience is in the center


	std::vector<ALuint> sources;
	std::vector<ALuint> buffers;
	std::vector<ALint> source_states;
	std::vector<int> bufferbound;

	size_t m_Nabctracks=0;

    std::vector<size_t> m_originalpartnumbers;

    std::vector< int > m_mutes; // internal information about tracks being muted

    int m_volume = 100;  // listener volume
    int m_panning = 100; // panning percentage, 100=full, 0 = all in center
    int m_durationseconds;

    // for the synchronization
    std::chrono::time_point<std::chrono::high_resolution_clock> m_ABC_Play_Start;       // this is when we started
    std::chrono::time_point<std::chrono::high_resolution_clock> m_ABC_Play_LastUpdate;  // this is where we are now
    std::thread * PlayThread;

    std::vector<uint64_t> trackpositions;



    std::vector<float> m_envelope; // constains the envelop multiply function to be used for sending the tone

    void SetEnvelope(int Instrument, uint32_t duration, uint32_t samplesize);



    std::vector<std::vector<int>> oggpitchnumbers=
    {
        {36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72},
        {36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72},
        {36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72},
        {36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72},
        {36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72},
        {36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72},
        {36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72},
        {36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72},
        {36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72},
        {36},
        {36},
        {36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72},
        {36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72},
        {36,37,38,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72},
        {36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72},
        {36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72},
        {36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72},
        {36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72},
        {36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72},
        {36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72},
        {36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72},
        {36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72},
        {36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72}
    };

    std::vector<float> fadeouts = {0,0,0,0.2,0.2,0.2,0.2,0.2,0,0  ,0  ,0  ,0  ,0.2  ,0.2  ,0.4, 0   ,0.2   ,0.2 ,0.2,  0.2,  0.5, 0};

    // necesarry definitions

    std::string lotroinstruments_formal[23] =
          {"Lute of the Ages", "Basic Harp", "Theorbo", "Horn", "Clarinet", "Flute", "Bagpipes", "Pipgorn", "Drums",
     //      0      1        2         3         4          5        6           7         8
           "Cowbell", "Moor Cowbell", "Basic Lute", "Misty M. Harp", "Student Fiddle", "Lonely M. Fiddle", "Sprightly Fiddle", "Travellers Tr.", "Bardic Fiddle",
     //      9          10                 11            12         13          14           15         16          17
           "Basic Fiddle", "Basic Basson", "Lonely M. Basson", "Brusque Basson", "Hand-Knells"};
     //      18             19              20            21


     
     std::vector<float> relativegain = {1, 1.25, 0.56, 1.25, 0.95, 0.95, 0.95, 0.95, 1.25, 1., 1.25, 0.55, 0.74, 1.  
                                           , 1.25, 0.6, 0.8, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.0};
     std::vector<float> pitchgains = {0.2, 0.2, 0.28, 0.37, 0.46, 0.55, 0.64, 0.73, 0.78, 0.78};




};


std::vector<std::vector<ToneTuple>> AudioPlayerAL::ToneData()
{ 
    return myabc->m_ABCTonesvector;
}

std::vector<size_t>AudioPlayerAL::GetOriginalPartNumbers(){
    return this->m_originalpartnumbers;
}


void AudioPlayerAL::ExportWAV(std::string * filename)
{
    WaitForSamples();

    // 1. Snapshot the data to avoid interference with the playback thread or SendABC
    m.lock();
    if (myabc == NULL || m_Nabctracks == 0) {
        m.unlock();
        return;
    }
    std::vector<std::vector<ToneTuple>> toneData = myabc->m_ABCTonesvector;
    int64_t durationSec = m_durationseconds;
    int globalPanning = m_panning;
    float globalVolume = m_volume * 0.01f;
    std::vector<int> trackPans;
    for (size_t i = 0; i < m_Nabctracks; i++) trackPans.push_back(myabc->GetStereoPosition(i));
    std::vector<int> trackMutes = m_mutes;
    m.unlock();

    // 2. Setup rendering buffers (Stereo 16-bit 44.1kHz)
    // We add a few seconds for safety to allow final note fade-outs to finish
    uint64_t totalSamples = (durationSec + 5) * 44100;
    std::vector<float> leftBuf(totalSamples, 0.0f);
    std::vector<float> rightBuf(totalSamples, 0.0f);

    // 3. Offline Render/Mix
    for (size_t i = 0; i < toneData.size(); i++)
    {
        if (i < trackMutes.size() && trackMutes[i] != 0) continue;

        // Calculate Stereo Panning factors
        // trackPanX represents the X-coordinate of the source, relative to the listener.
        // Its range is roughly [-0.48, 0.35] based on default3Dpositions and globalPanning.
        // We normalize this to a [-1.0, 1.0] range for the equal-power panning formula.
        float trackPanX = trackPans[i] * 0.01f * globalPanning * 0.01f; // This is the effective X-position
        float normalized_pan_pos = trackPanX * 0.5f; // Assuming max deviation is 0.5 for full pan
        normalized_pan_pos = std::clamp(normalized_pan_pos, -1.0f, 1.0f); // Clamp to ensure it's within [-1, 1]
        float panL = std::sqrt(0.5f * (1.0f - normalized_pan_pos));
        float panR = std::sqrt(0.5f * (1.0f + normalized_pan_pos));
        for (auto& tone : toneData[i])
        {
            int64_t startSample = std::get<0>(tone);
            int64_t duration = std::get<2>(tone);
            int instrument = std::get<3>(tone);
            int pitchIdx = std::get<4>(tone); // ToneTuple stores pitch+36 already
            int velocity = (int)std::get<5>(tone);

            // Cowbell and Moor Cowbell (instruments 9 and 10) always use sample index 36
            if (instrument == 9 || instrument == 10) pitchIdx = 36;

            if (instrument < 0 || instrument >= 23) continue;
            if (pitchIdx < 0 || pitchIdx >= (int)allsamples[instrument].size()) continue;

            const std::vector<uint8_t>& sourceData = allsamples[instrument][pitchIdx];
            if (sourceData.size() < 2) continue;

            float gain = AudioPlayerAL::relativegain[instrument] * AudioPlayerAL::pitchgains[std::max(0, std::min(velocity, 9))] * globalVolume;
            uint32_t fadesamples = static_cast<uint32_t>(AudioPlayerAL::fadeouts[instrument] * 44100);

            size_t mixSamples = 0;
            if (fadeouttype[instrument] == 0)
                mixSamples = sourceData.size() / 2;
            else
                mixSamples = std::min((size_t)(duration + fadesamples), sourceData.size() / 2);

            const int16_t* samples = (const int16_t*)sourceData.data();

            for (size_t s = 0; s < mixSamples; s++)
            {
                if (startSample + (int64_t)s >= (int64_t)totalSamples) break;

                float sampleVal = (float)samples[s] / 32768.0f;

                // Apply Fadeout multiplier matching PlayLoop logic
                if (fadeouttype[instrument] != 0 && s >= (size_t)duration && fadesamples > 0)
                {
                    float fadePos = (float)(s - duration);
                    sampleVal *= (1.0f - (fadePos / fadesamples));
                }

                leftBuf[startSample + s] += sampleVal * gain * panL;
                rightBuf[startSample + s] += sampleVal * gain * panR;
            }
        }
    }

    // 4. Output to WAV File
    std::ofstream file(*filename, std::ios::binary);
    if (!file.is_open()) return;

    uint32_t dataSize = (uint32_t)totalSamples * 2 * 2;
    uint32_t overallSize = 36 + dataSize;
    uint32_t fmtSize = 16;
    uint16_t formatType = 1; // PCM
    uint16_t channels = 2;
    uint32_t sampleRate = 44100;
    uint32_t byteRate = 44100 * 4;
    uint16_t blockAlign = 4;
    uint16_t bitsPerSample = 16;

    file.write("RIFF", 4); file.write((char*)&overallSize, 4); file.write("WAVE", 4);
    file.write("fmt ", 4); file.write((char*)&fmtSize, 4); file.write((char*)&formatType, 2);
    file.write((char*)&channels, 2); file.write((char*)&sampleRate, 4);
    file.write((char*)&byteRate, 4); file.write((char*)&blockAlign, 2);
    file.write((char*)&bitsPerSample, 2);
    file.write("data", 4); file.write((char*)&dataSize, 4);

    for (size_t i = 0; i < totalSamples; i++)
    {
        int16_t l = (int16_t)std::max(-32768.0f, std::min(32767.0f, leftBuf[i] * 32767.0f));
        int16_t r = (int16_t)std::max(-32768.0f, std::min(32767.0f, rightBuf[i] * 32767.0f));
        file.write((char*)&l, 2); file.write((char*)&r, 2);
    }
    file.close();
    std::cout << "ABC rendered to " << *filename << std::endl;
}




std::vector<float> AudioPlayerAL::GenerateWAVMonof()
{
    WaitForSamples();

    // 1. Snapshot and flatten active data under a brief lock
    std::vector<ToneTuple> flatTones;
    m.lock();
    if (myabc == nullptr || m_Nabctracks == 0) {
        m.unlock();
        return std::vector<float>();
    }

    int64_t durationSec = m_durationseconds;
    float globalVolume = m_volume * 0.01f;
    std::vector<int> trackMutes = m_mutes;

    size_t totalExpectedTones = 0;
    for (size_t i = 0; i < myabc->m_ABCTonesvector.size(); i++) {
        if (i >= trackMutes.size() || trackMutes[i] == 0) {
            totalExpectedTones += myabc->m_ABCTonesvector[i].size();
        }
    }
    flatTones.reserve(totalExpectedTones);

    for (size_t i = 0; i < myabc->m_ABCTonesvector.size(); i++)
    {
        if (i < trackMutes.size() && trackMutes[i] != 0) continue;
        const auto& trackTones = myabc->m_ABCTonesvector[i];
        flatTones.insert(flatTones.end(), trackTones.begin(), trackTones.end());
    }
    m.unlock(); // Lock released early

    // 2. Sort chronologically by startSample
    std::sort(flatTones.begin(), flatTones.end(), [](const ToneTuple& a, const ToneTuple& b) {
        return std::get<0>(a) < std::get<0>(b);
    });

    // 3. Setup Integer Mixing Buffer
    uint64_t totalSamples = (durationSec + 5) * 44100;
    std::vector<int32_t> mixBuf(totalSamples, 0); // Pure integer accumulation

    // 4. Offline Render/Mix (Pure Integer Operations)
    for (const auto& tone : flatTones)
    {
        int64_t startSample = std::get<0>(tone);
        int64_t duration = std::get<2>(tone);
        int instrument = std::get<3>(tone);
        int pitchIdx = std::get<4>(tone); 
        int velocity = static_cast<int>(std::get<5>(tone));

        if (instrument == 9 || instrument == 10) pitchIdx = 36;

        if (instrument < 0 || instrument >= 23) continue;
        if (pitchIdx < 0 || pitchIdx >= static_cast<int>(allsamples[instrument].size())) continue;

        const std::vector<uint8_t>& sourceData = allsamples[instrument][pitchIdx];
        if (sourceData.size() < 2) continue;

        // Convert gain components into a single fixed-point or scalar integer scale
        // Since velocity and relative gains vary, we pre-calculate a scalar float factor 
        // but apply it inside the integer math step to avoid full float buffer thrashing.
        float rawGain = AudioPlayerAL::relativegain[instrument] * 
                        AudioPlayerAL::pitchgains[std::max(0, std::min(velocity, 9))] * 
                        globalVolume;
        
        uint32_t fadesamples = static_cast<uint32_t>(AudioPlayerAL::fadeouts[instrument] * 44100.0f);
        size_t totalSourceSamples = sourceData.size() / 2; 

        size_t mixSamples = (fadeouttype[instrument] == 0) ? 
                             totalSourceSamples : 
                             std::min(static_cast<size_t>(duration + fadesamples), totalSourceSamples);

        if (startSample < 0 || static_cast<uint64_t>(startSample) >= totalSamples) continue;

        uint64_t max_allowed = totalSamples - static_cast<uint64_t>(startSample);
        uint64_t safe_looplimit = std::min(static_cast<uint64_t>(mixSamples), max_allowed);
        uint64_t safe_duration  = std::min(static_cast<uint64_t>(duration), safe_looplimit);

        const int16_t* __restrict src = reinterpret_cast<const int16_t*>(sourceData.data()); 
        int32_t* __restrict dest = &mixBuf[startSample];

        if (fadeouttype[instrument] == 0)
        {
            // LOOP 1: Blazing fast integer multiplication and accumulation
            for (uint64_t s = 0; s < safe_looplimit; ++s)
            {
                dest[s] += static_cast<int32_t>(src[s]) * rawGain; 
            }
        }
        else
        {
            // LOOP 2: Standard duration
            for (uint64_t s = 0; s < safe_duration; ++s)
            {
                dest[s] += static_cast<int32_t>(src[s]) * rawGain;
            }

            // LOOP 3: Fadeout section
            if (fadesamples > 0 && safe_looplimit > safe_duration)
            {
                const float ifadesamples = 1.0f / static_cast<float>(fadesamples);

                for (uint64_t s = safe_duration; s < safe_looplimit; ++s)
                {
                    float fadePos = static_cast<float>(s - safe_duration);
                    float fadeFactor = 1.0f - (fadePos * ifadesamples);
                    
                    dest[s] += static_cast<int32_t>(static_cast<float>(src[s]) * rawGain * fadeFactor);
                }
            }
        }
    }

    // 5. Final Single-Pass Conversion to Float & Normalization for STFT
    // Doing this at the very end in a linear array allows the compiler to use 
    // highly optimized SIMD instructions (like AVX2 _mm256_cvtepi32_ps) to convert 
    // 8 samples per instruction cycle.
    std::vector<float> rightBuf(totalSamples);
    const float normalizeFactor = 1.0f / 32768.0f;
    
    float* __restrict finalDest = rightBuf.data();
    const int32_t* __restrict intermediateSrc = mixBuf.data();

    #pragma omp simd
    for (uint64_t i = 0; i < totalSamples; ++i)
    {
        finalDest[i] = static_cast<float>(intermediateSrc[i]) * normalizeFactor;
    }

    return rightBuf;
}


std::vector<int32_t> AudioPlayerAL::GenerateWAVMonoI()
{
    WaitForSamples();

    m.lock();
    if (myabc == NULL || m_Nabctracks == 0) {
        m.unlock();
        return std::vector<int32_t>({});
    }
    std::vector<std::vector<ToneTuple>> toneData = myabc->m_ABCTonesvector;
    int64_t durationSec = m_durationseconds;
    float globalVolume = m_volume * 0.01f;
    std::vector<int> trackMutes = m_mutes;
    m.unlock();

    uint64_t totalSamples = (durationSec + 5) * 44100;
    std::vector<int32_t> rightBuf(totalSamples, 0);
    int32_t* __restrict bufData = rightBuf.data();

    for (size_t i = 0; i < toneData.size(); i++)
    {
        if (i < trackMutes.size() && trackMutes[i] != 0) continue;

        for (auto& tone : toneData[i])
        {
            int64_t startSample = std::get<0>(tone);
            int64_t duration = std::get<2>(tone);
            int instrument = std::get<3>(tone);
            int pitchIdx = std::get<4>(tone); 
            int velocity = (int)std::get<5>(tone);
         //   std::cout << "Full Velo Extracted " << velocity << std::endl;

            if (instrument == 9 || instrument == 10) pitchIdx = 36;
            if (instrument < 0 || instrument >= 23) continue;
            if (pitchIdx < 0 || pitchIdx >= (int)allsamples[instrument].size()) continue;

            const std::vector<uint8_t>& sourceData = allsamples[instrument][pitchIdx];
            if (sourceData.size() < 2) continue;

            float toneGain = AudioPlayerAL::relativegain[instrument] * 
                             AudioPlayerAL::pitchgains[std::max(0, std::min(velocity, 9))] * 
                             globalVolume;

         //   std::cout << " Full Generation " << toneGain << std::endl;
                             
            uint32_t fadesamples = static_cast<uint32_t>(AudioPlayerAL::fadeouts[instrument] * 44100);

            size_t mixSamples = (fadeouttype[instrument] == 0)
                ? (sourceData.size() / 2)
                : std::min(static_cast<size_t>(duration + fadesamples), sourceData.size() / 2);

            const int16_t* __restrict src = reinterpret_cast<const int16_t*>(sourceData.data());

            // FIX: Added guard for early boundaries matching ApplyToneDeltaMonoI
            if (startSample < 0 || static_cast<size_t>(startSample) >= totalSamples) continue;

            uint64_t looplimit = mixSamples;
            if (static_cast<size_t>(startSample) + mixSamples > totalSamples) {
                looplimit = totalSamples - static_cast<size_t>(startSample);
            }
            if (looplimit == 0) continue;

            int32_t* __restrict dest = &bufData[startSample];

            if (fadeouttype[instrument] == 0)
            {
                #pragma omp simd
                for (uint64_t s = 0; s < looplimit; ++s)
                {
                    dest[s] += static_cast<int32_t>(static_cast<float>(src[s]) * toneGain);
                }
            }
            else
            {
                // FIX: Guard loops identically to ApplyToneDeltaMonoI
                uint64_t first_limit = std::min(static_cast<uint64_t>(duration), looplimit);

                #pragma omp simd
                for (uint64_t s = 0; s < first_limit; ++s)
                {
                    dest[s] += static_cast<int32_t>(static_cast<float>(src[s]) * toneGain);
                }

                if (looplimit > first_limit && fadesamples > 0) 
                {
                    const float ifadesamples = 1.0f / fadesamples;
                    for (uint64_t s = first_limit; s < looplimit; s++)
                    {
                        float fadeMultiplier = 1.0f - (static_cast<float>(s - duration) * ifadesamples);
                        float finalGain = toneGain * std::max(0.0f, fadeMultiplier);
                        dest[s] += static_cast<int32_t>(static_cast<float>(src[s]) * finalGain);
                    }
                }
            }
        }
    }

    return rightBuf;
}

std::vector<int32_t> AudioPlayerAL::GenerateWAVMonoI_11()
{
    WaitForSamples();

    m.lock();
    if (myabc == NULL || m_Nabctracks == 0) {
        m.unlock();
        return std::vector<int32_t>({});
    }
    std::vector<std::vector<ToneTuple>> toneData = myabc->m_ABCTonesvector;
    int64_t durationSec = m_durationseconds;
    float globalVolume = m_volume * 0.01f;
    std::vector<int> trackMutes = m_mutes;
    m.unlock();

    // 1. Scale output sample layout down to 11025 Hz
    uint64_t totalSamples = (durationSec + 5) * 11025;
    std::vector<int32_t> rightBuf(totalSamples, 0);
    int32_t* __restrict bufData = rightBuf.data();

    for (size_t i = 0; i < toneData.size(); i++)
    {
        if (i < trackMutes.size() && trackMutes[i] != 0) continue;

        for (auto& tone : toneData[i])
        {
            // 2. Downscale the 44.1kHz timeline variables to 11.025kHz
            int64_t startSample = std::get<0>(tone) / 4;
            int64_t duration    = std::get<2>(tone) / 4;
            int instrument      = std::get<3>(tone);
            int pitchIdx        = std::get<4>(tone); 
            int velocity        = (int)std::get<5>(tone);

            if (instrument == 9 || instrument == 10) pitchIdx = 36;
            if (instrument < 0 || instrument >= 23) continue;
            
            // 3. Look up from your updated int16_t table version
            if (pitchIdx < 0 || pitchIdx >= (int)allsamples_11[instrument].size()) continue;

            const std::vector<uint8_t>& sourceData = allsamples_11[instrument][pitchIdx];
            if (sourceData.size() < 2) continue;

            float toneGain = AudioPlayerAL::relativegain[instrument] * 
                             AudioPlayerAL::pitchgains[std::max(0, std::min(velocity, 9))] * 
                             globalVolume;

            // 4. Scale fade-out parameters down to 11025 Hz
            uint32_t fadesamples = static_cast<uint32_t>(AudioPlayerAL::fadeouts[instrument] * 11025);

            // 5. Brought back the byte size division (/ 2) for int16_t layouts
            size_t mixSamples = (fadeouttype[instrument] == 0)
                ? (sourceData.size() / 2)
                : std::min(static_cast<size_t>(duration + fadesamples), sourceData.size() / 2);

            const int16_t* __restrict src = reinterpret_cast<const int16_t*>(sourceData.data());

            if (startSample < 0 || static_cast<size_t>(startSample) >= totalSamples) continue;

            uint64_t looplimit = mixSamples;
            if (static_cast<size_t>(startSample) + mixSamples > totalSamples) {
                looplimit = totalSamples - static_cast<size_t>(startSample);
            }
            if (looplimit == 0) continue;

            int32_t* __restrict dest = &bufData[startSample];

            if (fadeouttype[instrument] == 0)
            {
                // No 32768.0f multiplier needed since source and dest match integer scales

                /*
                #pragma omp simd
                for (uint64_t s = 0; s < looplimit; ++s)
                {
                    dest[s] += static_cast<int32_t>(static_cast<float>(src[s]) * toneGain);
                }*/

                #include "sampleaddloop.h"

            }
            else
            {
                uint64_t first_limit = std::min(static_cast<uint64_t>(duration), looplimit);

                #pragma omp simd
                for (uint64_t s = 0; s < first_limit; ++s)
                {
                    dest[s] += static_cast<int32_t>(static_cast<float>(src[s]) * toneGain);
                }

                if (looplimit > first_limit && fadesamples > 0) 
                {
                    const float ifadesamples = 1.0f / static_cast<float>(fadesamples);
                    for (uint64_t s = first_limit; s < looplimit; s++)
                    {
                        float fadeMultiplier = 1.0f - (static_cast<float>(s - duration) * ifadesamples);
                        float finalGain = toneGain * std::max(0.0f, fadeMultiplier);
                        dest[s] += static_cast<int32_t>(static_cast<float>(src[s]) * finalGain);
                    }
                }
            }
        }
    }

    return rightBuf;
}

std::vector<int32_t> AudioPlayerAL::ApplyToneDeltaMonoI(std::vector<int32_t> sampleBuf,
                                      const std::vector<ToneTuple>& added,
                                      const std::vector<ToneTuple>& deleted) {
    size_t currentSize = sampleBuf.size();
    size_t requiredMaxEnd = currentSize;

    // 1. Scan added tones to determine the absolute maximum required sample boundary
    for (const auto& tone : added) {
        int64_t startSample = std::get<0>(tone);
        int64_t duration    = std::get<2>(tone);
        int instrument      = std::get<3>(tone);
        int pitchIdx        = std::get<4>(tone);

        if (instrument == 9 || instrument == 10) pitchIdx = 36;
        if (instrument < 0 || instrument >= 23) continue;
        if (pitchIdx < 0 || pitchIdx >= (int)allsamples[instrument].size()) continue;

        const std::vector<uint8_t>& sourceData = allsamples[instrument][pitchIdx];
        if (sourceData.size() < 2) continue;

        uint32_t fadesamples = static_cast<uint32_t>(AudioPlayerAL::fadeouts[instrument] * 44100);

        size_t mixSamples = (fadeouttype[instrument] == 0)
            ? (sourceData.size() / 2)
            : std::min(static_cast<size_t>(duration + fadesamples), sourceData.size() / 2);

        if (startSample >= 0) {
            size_t requiredEnd = static_cast<size_t>(startSample) + mixSamples;
            if (requiredEnd > requiredMaxEnd) {
                requiredMaxEnd = requiredEnd;
            }
        }
    }

    if (requiredMaxEnd > currentSize) {
        size_t safetyPadding = 5 * 44100; 
        size_t targetSamples = requiredMaxEnd + safetyPadding;
        sampleBuf.resize(targetSamples, 0);
    }

    float globalVolume = m_volume * 0.01f;
    int32_t* __restrict bufData = sampleBuf.data();
    size_t totalSamples = sampleBuf.size(); 

    // 2. Sequential loops using identical variables, casting, and structure from GenerateWAVMonoI
    // Processing Deletions Pass
    for (const auto& tone : deleted) {
        int64_t startSample = std::get<0>(tone);
        int64_t duration = std::get<2>(tone);
        int instrument = std::get<3>(tone);
        int pitchIdx = std::get<4>(tone); 
        
        // Match the exact truncation casting style from GenerateWAVMonoI
        int velocity = (int)std::get<5>(tone) +1; 
       // std::cout << "Partial Velo Del Extracted " << velocity << std::endl;

        if (instrument == 9 || instrument == 10) pitchIdx = 36;
        if (instrument < 0 || instrument >= 23) continue;
        if (pitchIdx < 0 || pitchIdx >= (int)allsamples[instrument].size()) continue;

        const std::vector<uint8_t>& sourceData = allsamples[instrument][pitchIdx];
        if (sourceData.size() < 2) continue;

        float toneGain = AudioPlayerAL::relativegain[instrument] * 
                         AudioPlayerAL::pitchgains[std::max(0, std::min(velocity, 9))] * 
                         globalVolume * -1.0f;

                        // std::cout << " Partial Generation " << toneGain << std::endl;
                             
        uint32_t fadesamples = static_cast<uint32_t>(AudioPlayerAL::fadeouts[instrument] * 44100);

        size_t mixSamples = (fadeouttype[instrument] == 0)
            ? (sourceData.size() / 2)
            : std::min(static_cast<size_t>(duration + fadesamples), sourceData.size() / 2);

        const int16_t* __restrict src = reinterpret_cast<const int16_t*>(sourceData.data());

        if (startSample < 0 || static_cast<size_t>(startSample) >= totalSamples) continue;

        uint64_t looplimit = mixSamples;
        if (static_cast<size_t>(startSample) + mixSamples > totalSamples) {
            looplimit = totalSamples - static_cast<size_t>(startSample);
        }
        if (looplimit == 0) continue;

        int32_t* __restrict dest = &bufData[startSample];

        if (fadeouttype[instrument] == 0) {
            #pragma omp simd
            for (uint64_t s = 0; s < looplimit; ++s) {
                dest[s] += static_cast<int32_t>(static_cast<float>(src[s]) * toneGain);
            }
        } else {
            uint64_t first_limit = std::min(static_cast<uint64_t>(duration), looplimit);

            #pragma omp simd
            for (uint64_t s = 0; s < first_limit; ++s) {
                dest[s] += static_cast<int32_t>(static_cast<float>(src[s]) * toneGain);
            }

            if (looplimit > first_limit && fadesamples > 0) {
                const float ifadesamples = 1.0f / fadesamples;
                for (uint64_t s = first_limit; s < looplimit; s++) {
                    float fadeMultiplier = 1.0f - (static_cast<float>(s - duration) * ifadesamples);
                    float finalGain = toneGain * std::max(0.0f, fadeMultiplier);
                    dest[s] += static_cast<int32_t>(static_cast<float>(src[s]) * finalGain);
                }
            }
        }
    }

    // Processing Additions Pass
    for (const auto& tone : added) {
        int64_t startSample = std::get<0>(tone);
        int64_t duration = std::get<2>(tone);
        int instrument = std::get<3>(tone);
        int pitchIdx = std::get<4>(tone); 
        
        // Match the exact truncation casting style from GenerateWAVMonoI
        int velocity = (int)std::get<5>(tone) + 1; 

      //  std::cout << "Partial Velo Extracted " << velocity << std::endl;

        if (instrument == 9 || instrument == 10) pitchIdx = 36;
        if (instrument < 0 || instrument >= 23) continue;
        if (pitchIdx < 0 || pitchIdx >= (int)allsamples[instrument].size()) continue;

        const std::vector<uint8_t>& sourceData = allsamples[instrument][pitchIdx];
        if (sourceData.size() < 2) continue;

        float toneGain = AudioPlayerAL::relativegain[instrument] * 
                         AudioPlayerAL::pitchgains[std::max(0, std::min(velocity, 9))] * 
                         globalVolume;
        //                 std::cout << " Partial Generation Added " << toneGain << std::endl;
                             
        uint32_t fadesamples = static_cast<uint32_t>(AudioPlayerAL::fadeouts[instrument] * 44100);

        size_t mixSamples = (fadeouttype[instrument] == 0)
            ? (sourceData.size() / 2)
            : std::min(static_cast<size_t>(duration + fadesamples), sourceData.size() / 2);

        const int16_t* __restrict src = reinterpret_cast<const int16_t*>(sourceData.data());

        if (startSample < 0 || static_cast<size_t>(startSample) >= totalSamples) continue;

        uint64_t looplimit = mixSamples;
        if (static_cast<size_t>(startSample) + mixSamples > totalSamples) {
            looplimit = totalSamples - static_cast<size_t>(startSample);
        }
        if (looplimit == 0) continue;

        int32_t* __restrict dest = &bufData[startSample];

        if (fadeouttype[instrument] == 0) {
            #pragma omp simd
            for (uint64_t s = 0; s < looplimit; ++s) {
                dest[s] += static_cast<int32_t>(static_cast<float>(src[s]) * toneGain);
            }
        } else {
            uint64_t first_limit = std::min(static_cast<uint64_t>(duration), looplimit);

            #pragma omp simd
            for (uint64_t s = 0; s < first_limit; ++s) {
                dest[s] += static_cast<int32_t>(static_cast<float>(src[s]) * toneGain);
            }

            if (looplimit > first_limit && fadesamples > 0) {
                const float ifadesamples = 1.0f / fadesamples;
                for (uint64_t s = first_limit; s < looplimit; s++) {
                    float fadeMultiplier = 1.0f - (static_cast<float>(s - duration) * ifadesamples);
                    float finalGain = toneGain * std::max(0.0f, fadeMultiplier);
                    dest[s] += static_cast<int32_t>(static_cast<float>(src[s]) * finalGain);
                }
            }
        }
    }

    return sampleBuf;
}


std::vector<int32_t> AudioPlayerAL::ApplyToneDeltaMonoI_11(std::vector<int32_t> sampleBuf,
                                      const std::vector<ToneTuple>& added,
                                      const std::vector<ToneTuple>& deleted) {
    size_t currentSize = sampleBuf.size();
    size_t requiredMaxEnd = currentSize;

    // 1. Scan added tones to determine the absolute maximum required sample boundary
    for (const auto& tone : added) {
        // Scale timeline coordinates from 44.1kHz down to 11.025kHz
        int64_t startSample = std::get<0>(tone) / 4;
        int64_t duration    = std::get<2>(tone) / 4;
        int instrument      = std::get<3>(tone);
        int pitchIdx        = std::get<4>(tone);

        if (instrument == 9 || instrument == 10) pitchIdx = 36;
        if (instrument < 0 || instrument >= 23) continue;
        if (pitchIdx < 0 || pitchIdx >= (int)allsamples_11[instrument].size()) continue;

        // Uses your updated int16_t storage vector layout
        const std::vector<uint8_t>& sourceData = allsamples_11[instrument][pitchIdx];
        if (sourceData.size() < 2) continue;

        // Scale fade metrics down to 11025 Hz
        uint32_t fadesamples = static_cast<uint32_t>(AudioPlayerAL::fadeouts[instrument] * 11025);

        // Preserves your byte size sample counting (/ 2) for int16_t layouts
        size_t mixSamples = (fadeouttype[instrument] == 0)
            ? (sourceData.size() / 2)
            : std::min(static_cast<size_t>(duration + fadesamples), sourceData.size() / 2);

        if (startSample >= 0) {
            size_t requiredEnd = static_cast<size_t>(startSample) + mixSamples;
            if (requiredEnd > requiredMaxEnd) {
                requiredMaxEnd = requiredEnd;
            }
        }
    }

    if (requiredMaxEnd > currentSize) {
        // Updated safety buffer from 5 seconds * 44100 down to 5 seconds * 11025
        size_t safetyPadding = 5 * 11025; 
        size_t targetSamples = requiredMaxEnd + safetyPadding;
        sampleBuf.resize(targetSamples, 0);
    }

    float globalVolume = m_volume * 0.01f;
    int32_t* __restrict bufData = sampleBuf.data();
    size_t totalSamples = sampleBuf.size(); 

    // 2. Processing Deletions Pass
    for (const auto& tone : deleted) {
        int64_t startSample = std::get<0>(tone) / 4;
        int64_t duration    = std::get<2>(tone) / 4;
        int instrument      = std::get<3>(tone);
        int pitchIdx        = std::get<4>(tone); 
        int velocity        = (int)std::get<5>(tone) + 1; 

        if (instrument == 9 || instrument == 10) pitchIdx = 36;
        if (instrument < 0 || instrument >= 23) continue;
        if (pitchIdx < 0 || pitchIdx >= (int)allsamples_11[instrument].size()) continue;

        const std::vector<uint8_t>& sourceData = allsamples_11[instrument][pitchIdx];
        if (sourceData.size() < 2) continue;

        float toneGain = AudioPlayerAL::relativegain[instrument] * 
                         AudioPlayerAL::pitchgains[std::max(0, std::min(velocity, 9))] * 
                         globalVolume * -1.0f; // Subtract sound energy
                             
        uint32_t fadesamples = static_cast<uint32_t>(AudioPlayerAL::fadeouts[instrument] * 11025);

        size_t mixSamples = (fadeouttype[instrument] == 0)
            ? (sourceData.size() / 2)
            : std::min(static_cast<size_t>(duration + fadesamples), sourceData.size() / 2);

        const int16_t* __restrict src = reinterpret_cast<const int16_t*>(sourceData.data());

        if (startSample < 0 || static_cast<size_t>(startSample) >= totalSamples) continue;

        uint64_t looplimit = mixSamples;
        if (static_cast<size_t>(startSample) + mixSamples > totalSamples) {
            looplimit = totalSamples - static_cast<size_t>(startSample);
        }
        if (looplimit == 0) continue;

        int32_t* __restrict dest = &bufData[startSample];

        if (fadeouttype[instrument] == 0) {

            #include "sampleaddloop.h"

            /*
            #pragma omp simd
            for (uint64_t s = 0; s < looplimit; ++s) {
                dest[s] += static_cast<int32_t>(static_cast<float>(src[s]) * toneGain);
            }*/


        } else {
            uint64_t first_limit = std::min(static_cast<uint64_t>(duration), looplimit);

            #pragma omp simd
            for (uint64_t s = 0; s < first_limit; ++s) {
                dest[s] += static_cast<int32_t>(static_cast<float>(src[s]) * toneGain);
            }

            if (looplimit > first_limit && fadesamples > 0) {
                const float ifadesamples = 1.0f / static_cast<float>(fadesamples);
                for (uint64_t s = first_limit; s < looplimit; s++) {
                    float fadeMultiplier = 1.0f - (static_cast<float>(s - duration) * ifadesamples);
                    float finalGain = toneGain * std::max(0.0f, fadeMultiplier);
                    dest[s] += static_cast<int32_t>(static_cast<float>(src[s]) * finalGain);
                }
            }
        }
    }

    // 3. Processing Additions Pass
    for (const auto& tone : added) {
        int64_t startSample = std::get<0>(tone) / 4;
        int64_t duration    = std::get<2>(tone) / 4;
        int instrument      = std::get<3>(tone);
        int pitchIdx        = std::get<4>(tone); 
        int velocity        = (int)std::get<5>(tone) + 1; 

        if (instrument == 9 || instrument == 10) pitchIdx = 36;
        if (instrument < 0 || instrument >= 23) continue;
        if (pitchIdx < 0 || pitchIdx >= (int)allsamples_11[instrument].size()) continue;

        const std::vector<uint8_t>& sourceData = allsamples_11[instrument][pitchIdx];
        if (sourceData.size() < 2) continue;

        float toneGain = AudioPlayerAL::relativegain[instrument] * 
                         AudioPlayerAL::pitchgains[std::max(0, std::min(velocity, 9))] * 
                         globalVolume; // Add sound energy
                             
        uint32_t fadesamples = static_cast<uint32_t>(AudioPlayerAL::fadeouts[instrument] * 11025);

        size_t mixSamples = (fadeouttype[instrument] == 0)
            ? (sourceData.size() / 2)
            : std::min(static_cast<size_t>(duration + fadesamples), sourceData.size() / 2);

        const int16_t* __restrict src = reinterpret_cast<const int16_t*>(sourceData.data());

        if (startSample < 0 || static_cast<size_t>(startSample) >= totalSamples) continue;

        uint64_t looplimit = mixSamples;
        if (static_cast<size_t>(startSample) + mixSamples > totalSamples) {
            looplimit = totalSamples - static_cast<size_t>(startSample);
        }
        if (looplimit == 0) continue;

        int32_t* __restrict dest = &bufData[startSample];

        if (fadeouttype[instrument] == 0) {

            #include "sampleaddloop.h"
            /*
            #pragma omp simd
            for (uint64_t s = 0; s < looplimit; ++s) {
                dest[s] += static_cast<int32_t>(static_cast<float>(src[s]) * toneGain);
            }*/
        } else {
            uint64_t first_limit = std::min(static_cast<uint64_t>(duration), looplimit);

            #pragma omp simd
            for (uint64_t s = 0; s < first_limit; ++s) {
                dest[s] += static_cast<int32_t>(static_cast<float>(src[s]) * toneGain);
            }

            if (looplimit > first_limit && fadesamples > 0) {
                const float ifadesamples = 1.0f / static_cast<float>(fadesamples);
                for (uint64_t s = first_limit; s < looplimit; s++) {
                    float fadeMultiplier = 1.0f - (static_cast<float>(s - duration) * ifadesamples);
                    float finalGain = toneGain * std::max(0.0f, fadeMultiplier);
                    dest[s] += static_cast<int32_t>(static_cast<float>(src[s]) * finalGain);
                }
            }
        }
    }

    return sampleBuf;
}

std::vector<float> AudioPlayerAL::GenerateWAVMono()
{
    WaitForSamples();

    // 1. Snapshot the data to avoid interference with the playback thread or SendABC
    m.lock();
    if (myabc == NULL || m_Nabctracks == 0) {
        m.unlock();
        return std::vector<float>( {} );
    }
    std::vector<std::vector<ToneTuple>> toneData = myabc->m_ABCTonesvector;
    int64_t durationSec = m_durationseconds;
    int globalPanning = 0;
    float globalVolume = m_volume * 0.01f;

    std::vector<int> trackMutes = m_mutes;
    m.unlock();

    // 2. Setup rendering buffers (Stereo 16-bit 44.1kHz)
    // We add a few seconds for safety to allow final note fade-outs to finish
    uint64_t totalSamples = (durationSec + 5) * 44100;
    std::vector<float> rightBuf(totalSamples, 0.0f);

    // 3. Offline Render/Mix
    for (size_t i = 0; i < toneData.size(); i++)
    {
        if (i < trackMutes.size() && trackMutes[i] != 0) continue;

        // Calculate Stereo Panning factors
        // trackPanX represents the X-coordinate of the source, relative to the listener.
        // Its range is roughly [-0.48, 0.35] based on default3Dpositions and globalPanning.
        // We normalize this to a [-1.0, 1.0] range for the equal-power panning formula.

        for (auto& tone : toneData[i])
        {
            int64_t startSample = std::get<0>(tone);
            int64_t duration = std::get<2>(tone);
            int instrument = std::get<3>(tone);
            int pitchIdx = std::get<4>(tone); // ToneTuple stores pitch+36 already
            int velocity = (int)std::get<5>(tone);

            // Cowbell and Moor Cowbell (instruments 9 and 10) always use sample index 36
            if (instrument == 9 || instrument == 10) pitchIdx = 36;

            if (instrument < 0 || instrument >= 23) continue;
            if (pitchIdx < 0 || pitchIdx >= (int)allsamples[instrument].size()) continue;

            const std::vector<uint8_t>& sourceData = allsamples[instrument][pitchIdx];
            if (sourceData.size() < 2) continue;

            float gain = AudioPlayerAL::relativegain[instrument] * AudioPlayerAL::pitchgains[std::max(0, std::min(velocity, 9))] * globalVolume;
            uint32_t fadesamples = static_cast<uint32_t>(AudioPlayerAL::fadeouts[instrument] * 44100);

            size_t mixSamples = 0;
            if (fadeouttype[instrument] == 0)
                mixSamples = sourceData.size() / 2;
            else
                mixSamples = std::min((size_t)(duration + fadesamples), sourceData.size() / 2);

            const int16_t* samples = (const int16_t*)sourceData.data();


            uint64_t looplimit = mixSamples;
            if ( (uint64_t)startSample + (uint64_t)mixSamples > (int64_t)totalSamples ) looplimit = totalSamples - startSample;
            // the limit can not be beyond the end of the totalSamples .. just small sanity check


            if ( fadeouttype[instrument] == 0)
            {
                // up to the duration it is a simple loop:
                for (uint64_t s = 0; s < looplimit; ++s)
                {
                   float sampleVal = (float)samples[s] / 32768.0f;
                   rightBuf[startSample + s] += sampleVal * gain;
                }
            }
            else{

               // up to the duration it is a simple loop:
               
               const float allgain = gain / 32768.0f;
               const int16_t* __restrict src = &samples[0]; 
               float * __restrict dest = &rightBuf[startSample];

               //#pragma GCC unroll 4
               #pragma omp simd
               for (uint64_t s = 0; s < duration; ++s)
               {
                   dest[s] += (float)src[s] * allgain;
               }

               const float ifadesamples = 1.0 / fadesamples;

               //const float A = ( 1.0f - duration * ifadesamples)*allgain ;
               //const float B = ifadesamples * allgain;

               for (uint64_t s = duration; s < looplimit; s++)
               {
                   // if (startSample + (int64_t)s >= (int64_t)totalSamples) break;
                   float sampleVal = (float)samples[s] / 32768.0f;
                   
                   // Apply Fadeout multiplier matching PlayLoop logic

                   sampleVal *= (1.0f - ((float)(s - duration)) * ifadesamples);
                   
                   dest[s] += sampleVal * gain;
               }


            }

        }
    }

    return rightBuf;
}




void findAddedAndDeleted(
    const std::vector<ToneTuple>& A,
    const std::vector<ToneTuple>& B,
    std::vector<ToneTuple>& added,
    std::vector<ToneTuple>& deleted)
{
    // Create sorted copies to avoid modifying the original vectors
    std::vector<ToneTuple> sortedA = A;
    std::vector<ToneTuple> sortedB = B;

    std::sort(sortedA.begin(), sortedA.end());
    std::sort(sortedB.begin(), sortedB.end());

    // Reserve to avoid reallocations
    added.reserve(B.size());
    deleted.reserve(A.size());

    // Deleted: present in A but missing in B
    std::set_difference(sortedA.begin(), sortedA.end(),
                        sortedB.begin(), sortedB.end(),
                        std::back_inserter(deleted));

    // Added: present in B but missing in A
    std::set_difference(sortedB.begin(), sortedB.end(),
                        sortedA.begin(), sortedA.end(),
                        std::back_inserter(added));
}



ABCInput * AudioPlayerAL::GetABC()
{
    return myabc;
}

size_t AudioPlayerAL::GetNumberOfTracks()
{
    return m_Nabctracks;
}

int AudioPlayerAL::GetID(size_t track)
{
    // return m_id[track];
    return myabc->GetID(track);
}

int AudioPlayerAL::GetInstrument(size_t track)
{
   // return m_instrumentnumber[track];
   return myabc->GetInstrument(track);
}

int AudioPlayerAL::GetXNumber(size_t track)
{
    //return m_Xnumber[track];
    return myabc->GetX(track);
}

int AudioPlayerAL::GetPanning(size_t track)
{
    return myabc->GetStereoPosition(track);
}

int AudioPlayerAL::GetZPanning(size_t track)
{
    //return m_WavZPannings[track];
    return myabc->GetDepthPosition(track);
}

void AudioPlayerAL::UpdateABC(std::stringstream * abctext)
{

    Stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    SendABC(abctext);

}

void AudioPlayerAL::SetInstrument(int id, int instrument)
{
    // search through tracks to find ID
    for (size_t i = 0; i < m_Nabctracks; i++)
    {
        if (id == myabc->GetID(i))
        {
            // this is the one so we have to change all the instruments
            for (size_t j = 0; j < myabc->m_ABCTonesvector[i].size(); j++)
            {
               std::get<3>(myabc->m_ABCTonesvector[i][j]) = instrument;
            }
        }
    }
}

void AudioPlayerAL::SetMute(int id, bool value)
{
    // search through tracks to find ID
    for (size_t i = 0; i < m_Nabctracks; i++)
    {
        if (id == myabc->GetID(i))
        {
            // this is the one so we have to change all the instruments
            m_mutes[i] = value;
        }
    }
}

void AudioPlayerAL::SetPanning(int id, int panning)
{
    // search through tracks to find ID
    for (size_t i = 0; i < m_Nabctracks; i++)
    {
        if (id == myabc->GetID(i))
        {
            // this is the one so we have to change all the instruments
           // m_WavPannings[i] = panning;
            myabc->SetStereoPosition(i, panning);
        }
    }
}

void AudioPlayerAL::SetVolume(float value)
{
    m_volume = value;
    alListenerf(AL_GAIN, 1.0f * value/100.0f);
}

float AudioPlayerAL::GetVolume()
{
    return m_volume;
}

void AudioPlayerAL::SetGlobalPanning(int value)
{
    m_panning = value;
}

void AudioPlayerAL::ExportSamples()
{
    WaitForSamples();
    // std::ofstream
    for (size_t i = 0; i < allsamples.size(); i++ )
    {
        for (size_t j = 0; j < allsamples[i].size(); j++)
        {
            std::stringstream filename;
            filename << "samples/sample_" << i << "_" << j;
            std::ofstream outfile;
            outfile.open(filename.str(), std::ofstream::out);

            for ( size_t m = 0; m < allsamples[i][j].size()/2; m = m+1 )
            {
                short * value;
                value = (short*)(&allsamples[i][j][m*2]);
                outfile << m << "  " << value[0] << std::endl;
            }
            outfile.close();
        }
    }
}

void AudioPlayerAL::Play()
{
    m_ABC_Play_Start = std::chrono::high_resolution_clock::now();
    m_ABC_Play_LastUpdate = m_ABC_Play_Start;

    trackpositions.resize(m_Nabctracks);
    std::fill( trackpositions.begin(), trackpositions.end(), 0 );

    PlayThread = new std::thread(&AudioPlayerAL::PlayLoop, this);
    m_stop = 0;

}

void AudioPlayerAL::Stop()
{
   m_stop = 1;
}

void AudioPlayerAL::Seek(float f)
{
   m.lock();
   std::chrono::time_point<std::chrono::high_resolution_clock> thisisnow = std::chrono::high_resolution_clock::now();
   thisisnow -= std::chrono::milliseconds(static_cast<uint32_t>(f * m_durationseconds*1000));
   m_mute = 1;
   // we want to jump to a time that corresponds to now - f * duration
   m_ABC_Play_Start = thisisnow;
   m_ABC_Play_LastUpdate = std::chrono::high_resolution_clock::now();

   // now we need to set the pointers to the right position

   std::chrono::duration<double> ST = m_ABC_Play_LastUpdate - m_ABC_Play_Start;
   uint64_t st = uint64_t(ST.count()) * uint64_t(44100);  // Starting Time in samples
   for (size_t i = 0; i < m_Nabctracks; i++)
   {
      // we skip ahead in time
      size_t mytrackposition = 0;
      while (( mytrackposition < myabc->m_ABCTonesvector[i].size()  ) && ( static_cast<uint64_t>(std::get<0>(myabc->m_ABCTonesvector[i][mytrackposition])) < st  ))
      {
         mytrackposition++;
      }
      trackpositions[i] = mytrackposition;
   }

   // Stop currently playing sources so old notes don't linger
   for (size_t i = 0; i < 64; i++) alSourceStop(sources[i]);

   m_mute = 0;
   m.unlock();
}

float AudioPlayerAL::Position()
{
    if (m_stop || myabc == NULL || trackpositions.empty())
        return 0.0f;

    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - m_ABC_Play_Start;
    float pos = static_cast<float>(elapsed.count());

    if (pos < 0.0f) pos = 0.0f;
    if (pos > (float)m_durationseconds) pos = (float)m_durationseconds;

    return pos;
}

bool AudioPlayerAL::Finished()
{
   size_t trackfinished = 0;
   for (size_t i = 0; i < m_Nabctracks; i++)
   {
      if ( trackpositions[i] == myabc->m_ABCTonesvector[i].size() )
         trackfinished++;
   }
   if ( trackfinished == m_Nabctracks)
   {
       return true;
   }
   else
   {
       return false;
   }
}

void AudioPlayerAL::PlayLoop()
{
    WaitForSamples();
    // read current time
    m_ABC_Play_Start = std::chrono::high_resolution_clock::now();
    m_ABC_Play_LastUpdate = m_ABC_Play_Start;

    trackpositions.resize(m_Nabctracks);
    std::fill( trackpositions.begin(), trackpositions.end(), 0 );


    while (m_stop == 0)
    {

       // Get current time
       std::chrono::time_point<std::chrono::high_resolution_clock> updatetime = std::chrono::high_resolution_clock::now();

       // Delta Time from last update and from playstart
       std::chrono::duration<double> DT = updatetime - m_ABC_Play_LastUpdate;
       std::chrono::duration<double> ST = updatetime - m_ABC_Play_Start;

       uint64_t st = ST.count() * uint64_t(44100);  // Starting Time in samples
       uint64_t dt = DT.count() * uint64_t(44100);  // Delta Time in samples

       for (size_t i = 0; i < m_Nabctracks; i++)
       {
           // find the tones we need to send per track
           // if (std::get<0>(m_ABCTonesvector[i][trackpositions[i]]) >= st)

           // we take the next couple of tones in this track that had to be played
           while ((trackpositions[i] < myabc->m_ABCTonesvector[i].size()) && ( std::get<0>(myabc->m_ABCTonesvector[i][trackpositions[i]]) < static_cast<int64_t>(st+dt)  ))
           {
               int dontplay = 0;
               int instrument = std::get<3>(myabc->m_ABCTonesvector[i][trackpositions[i]]);
               int pitch = std::get<4>(myabc->m_ABCTonesvector[i][trackpositions[i]]) - 36;

               if (instrument == 10) pitch = 0;  // Cowbell and Moor Cowbell always play the 0 sample
               if (instrument == 9) pitch = 0;

               // We can only play in the existing range
               if (pitch < 0) dontplay = 1;
               if (pitch > 36) dontplay = 1;

               if (instrument == 13) {      // student fiddle has some additional empty tones
                  if ((pitch > 38) && (pitch < 43)) dontplay = 1;
               } 

               int velocity = std::get<5>(myabc->m_ABCTonesvector[i][trackpositions[i]]);
               if (velocity < 0) velocity = 0;
               if (velocity > 9) velocity = 9;
               size_t duration = std::get<2>(myabc->m_ABCTonesvector[i][trackpositions[i]]);

               m.lock();
               trackpositions[i]++;   // next tone
               m.unlock();

               if ((m_mute == 0) && ( m_mutes[i] == 0) && (dontplay==0))
               {
                   // find first free slot
                   size_t ii = 0;

                   ALint sourcestate;
                   alGetSourcei(sources[0], AL_SOURCE_STATE, &sourcestate);

                   while (( sourcestate == AL_PLAYING ) && ( ii < 63 ))
                   {
                       ii++;
                       alGetSourcei(sources[ii], AL_SOURCE_STATE, &sourcestate);
                   }
                   //  std::cout << " Source " << i << " not playing " << std::endl;
                   // If there is no free slot the sample will not be played - this maybe different in the game
                   if ( ii < 64)
                   {
                       std::vector<uint8_t> thissample;
                       uint32_t fadesamples = static_cast<uint32_t>(AudioPlayerAL::fadeouts[instrument] * 44100);
                       size_t mysize = 0;
                       if ( fadeouttype[instrument] == 0) // constant duration gets send as a whole
                       {
                           mysize = allsamples[instrument][pitch+36].size();  // size is always in 8bit units even if singular samples are 2 byte
                         //  std::cout << " Sample Size " << mysize << std::endl;
                       }
                       else
                       {
                           mysize = duration * 2 + fadesamples*2;  // size is memory for this sample in bytes
                        //   std::cout << " Sample Size Asked " << mysize << std::endl;
                           if (mysize > allsamples[instrument][pitch+36].size() ) mysize = allsamples[instrument][pitch+36].size();
                         //  std::cout << "Sample Size Got " << mysize << std::endl;
                             // if the sample is over then it is over
                       }
                       thissample.resize(mysize); // copy over all the data into thissample
                       for (size_t ij = 0; ij < mysize; ij++) thissample[ij] = allsamples[instrument][pitch+36][ij];


                       // we got the sample, now make sure we do the fadeout
                       // SetEnvelope(instrument, 1.0, allsamples[instrument][pitch+36].size()/2);

                       short * modulator = (short*)( &thissample[0] );
                       size_t fstart = thissample.size()/2-fadesamples;
                       size_t fend = thissample.size()/2;

                     //  std::cout << fadesamples << "   " << fstart << "   " << fend << std::endl;

                       for (size_t ij = fstart; ij < fend; ij++)  // we move from the regular play time end till the fadeout end
                       {
                           modulator[ij] = (short)(  ( modulator[ij] * (  (fend-ij)/(1.0f*fadesamples)  ) )+0.5 );
                       }


                       // was this buffer used before, if yes we have to do some stuff first?
                       if ( bufferbound[ii] == 1)
                       {
                            alSourceStop(sources[ii]);
                            alSourcei(sources[ii], AL_BUFFER, AL_NONE);
                            bufferbound[ii] = 0;
                       }

                       alBufferData(buffers[ii], AL_FORMAT_MONO16, &thissample[0], thissample.size(), 44100);
                   //    std::cout << "Dur " << duration << "  " << allsamples[instrument][pitch+36].size() << std::endl;
	                   alSourcei(sources[ii], AL_BUFFER, buffers[ii]);
	                   bufferbound[ii] = 1;

	                  // float mygain = fullvolumegains[instrument][velocity];
	                   float mygain = AudioPlayerAL::relativegain[instrument] * AudioPlayerAL::pitchgains[velocity] ;
                       alSourcef(sources[ii], AL_GAIN, mygain);

                       //float myp = m_WavPannings[i]*0.01 * m_panning*0.01;
                       float myp = myabc->GetStereoPosition(i) * 0.01 * m_panning * 0.01;

                    //   std::cout << "Setting Tone at " << i << " Panning " << m_WavPannings[i] << "  " << myp << "  Time " << std::get<0>(m_ABCTonesvector[i][trackpositions[i]])* 1.0f/(44100) << std::endl;



                       alSource3f(sources[ii], AL_POSITION, myp, 0, 0);

                   //    std::cout << "Playing Source " << i << " Instrument " << instrument << " Pitch " << pitch << " Gain " << mygain << " Samplesize " << allsamples[instrument][pitch+36].size() << std::endl;
	                   alSourcePlay(sources[ii]);
	                  // std::this_thread::sleep_for(std::chrono::milliseconds(1));

                   }
                   //else{std::cout << "Dropping Tone" << std::endl;}
               }
           }
       }


       // Wait a little and recall this routine if we are not supposed to stop yet
       m_ABC_Play_LastUpdate = updatetime;
       std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

AudioPlayerAL::~AudioPlayerAL()
{
    if (m_loadingThread && m_loadingThread->joinable()) {
        m_loadingThread->join();
    }
    delete m_loadingThread;

    alDeleteSources(64, &sources[0]);
	alDeleteBuffers(64, &buffers[0]);
	device = alcGetContextsDevice(context);
	alcMakeContextCurrent(NULL);
	alcDestroyContext(context);
	alcCloseDevice(device);
    if ( &OV_CALLBACKS_DEFAULT == &OV_CALLBACKS_NOCLOSE ) {};
    if ( &OV_CALLBACKS_STREAMONLY_NOCLOSE == &OV_CALLBACKS_STREAMONLY ) {};
}

void AudioPlayerAL::WaitForSamples()
{
    if (m_samplesLoaded.load()) return;
    std::unique_lock<std::mutex> lock(m_loadingMutex);
    m_loadingCV.wait(lock, [this] { return m_samplesLoaded.load(); });
}

void AudioPlayerAL::Initialize(float volume, int panning)
{
    m_samplesLoaded = false;

    // Allocate Sample Space
    allsamples.resize(AudioPlayerAL::oggpitchnumbers.size());
    allsamples_11.resize(AudioPlayerAL::oggpitchnumbers.size());


	for (size_t i = 0; i < AudioPlayerAL::oggpitchnumbers.size(); i++)
	{
	    allsamples[i].resize(73); // yes we also allocate 0s for the pitches below the ones we actually use ...
        allsamples_11[i].resize(73);
	    for (size_t j = 0; j < AudioPlayerAL::oggpitchnumbers[i].size(); j++)
        {
            int mypoint = AudioPlayerAL::oggpitchnumbers[i][j];
            allsamples[i][mypoint].resize(1);
            allsamples_11[i][mypoint].resize(1);
        }
	}

    // Load and Decompress the Samples in background
    m_loadingThread = new std::thread(&AudioPlayerAL::LoadSamplesBackground, this);

    // Check for Enumeration
    enumeration = alcIsExtensionPresent(NULL, "ALC_ENUMERATION_EXT");

	if (enumeration == AL_FALSE)
		fprintf(stderr, "enumeration extension not available\n");
    else std::cout << "Enumeration possible" << std::endl;

    // Allocate OpenAl Device
    device = alcOpenDevice(NULL);
    if (!device) {
        fprintf(stderr, "Error: Could not open default OpenAL device\n");
        {
            std::lock_guard<std::mutex> lock(m_loadingMutex);
            m_samplesLoaded = true;
        }
        m_loadingCV.notify_all();
        return;
    }
    std::cout << "Using " << alcGetString(device, ALC_DEVICE_SPECIFIER) << std::endl;

    if (alcIsExtensionPresent(device, "ALC_EXT_EFX") == AL_FALSE)
        std::cout << " no effects possible" << std::endl;
    else{ std::cout << "effects possible" << std::endl;}

    if(alcIsExtensionPresent(NULL, "ALC_SOFT_loopback"))
    {
        std::cout << "Loopback supported" << std::endl;
    }

    ALint srate;
    alcGetIntegerv(device, ALC_FREQUENCY, 1, &srate);
    std::cout << "OpenAL Frequency: " << srate << std::endl;

    // Create the OpenAl Context
    context = alcCreateContext(device, NULL);
    if (!context) {
        fprintf(stderr, "Error: Could not create OpenAL context\n");
        return;
    }

    if (!alcMakeContextCurrent(context)) {
		fprintf(stderr, "failed to make default context\n");
        return;
	}

	// Set the listener to 0 by default
    alListener3f(AL_POSITION, 0, 0, -1.0f);
	alListener3f(AL_VELOCITY, 0, 0, 0);
	alListenerfv(AL_ORIENTATION, listenerOri);
	//alListenerf(AL_GAIN, 0.985f);
    m_volume = static_cast<int>(volume);
    alListenerf(AL_GAIN, volume / 100.0f);

	// Allocate and initialize all 64 sources
    sources.resize(64);
	alGenSources((ALuint)64, &sources[0]);
	for (size_t i = 0; i < sources.size(); i ++)
    {
	   alSourcef(sources[i], AL_PITCH, 1);
	   alSourcef(sources[i], AL_GAIN, 1);
	   alSource3f(sources[i], AL_POSITION, 0, 0, 0);
	   alSource3f(sources[i], AL_VELOCITY, 0, 0, 0);
	   alSourcei(sources[i], AL_LOOPING, AL_FALSE);
    }

    // we use one buffer per source .. as simple as possible
    buffers.resize(64);
    alGenBuffers((ALuint)64, &buffers[0]);
    bufferbound.resize(64);
    std::fill(bufferbound.begin(), bufferbound.end(), 0);

    source_states.resize(64);
    m_panning = panning;
}

struct ogg_memory_buffer {
    const uint8_t* data;
    size_t size;
    size_t pos;
};

static size_t ov_read_mem(void* ptr, size_t size, size_t nmemb, void* datasource) {
    ogg_memory_buffer* buffer = (ogg_memory_buffer*)datasource;
    size_t total_size = size * nmemb;
    if (buffer->pos + total_size > buffer->size)
        total_size = buffer->size - buffer->pos;
    memcpy(ptr, buffer->data + buffer->pos, total_size);
    buffer->pos += total_size;
    return total_size / size;
}

static int ov_seek_mem(void* datasource, ogg_int64_t offset, int whence) {
    ogg_memory_buffer* buffer = (ogg_memory_buffer*)datasource;
    switch (whence) {
        case SEEK_SET: buffer->pos = (size_t)offset; break;
        case SEEK_CUR: buffer->pos += (size_t)offset; break;
        case SEEK_END: buffer->pos = buffer->size + (size_t)offset; break;
        default: return -1;
    }
    return 0;
}

static long ov_tell_mem(void* datasource) {
    return (long)((ogg_memory_buffer*)datasource)->pos;
}

static int ov_close_mem(void* datasource) {
    return 0;
}

std::vector<uint8_t> AudioPlayerAL::snd_load_file(const uint8_t* buffer, size_t size) {

    ogg_memory_buffer buffer_context = { buffer, size, 0 };
    ov_callbacks callbacks = { ov_read_mem, ov_seek_mem, ov_close_mem, ov_tell_mem };

	OggVorbis_File  oggStream;
	vorbis_info*    vorbisInfo;

	//vorbis_comment* vorbisComment;

	ALenum format;
	uint32_t BUFFER_SIZE = 8*4096;

	int result = ov_open_callbacks(&buffer_context, &oggStream, NULL, 0, callbacks);
	std::vector<uint8_t> outdata; // Moved declaration to here
    if (result < 0) {
        fprintf(stderr, "Error: Failed to open Ogg callbacks (result: %d)\n", result);
        return outdata;
    }

	vorbisInfo = ov_info(&oggStream, -1);
	// vorbisComment = ov_comment(&oggStream, -1);
	if(vorbisInfo->channels == 1)
		format = AL_FORMAT_MONO16;
	else
		format = AL_FORMAT_STEREO16;
    if (format == AL_FORMAT_MONO16) {};

//	char * dyn_data = NULL;
	int  mysize = 0;
	int  section;
	result = 1; // printf("Loading sound file\n");

	while(result > 0){

		char data[BUFFER_SIZE];

        //char * data;
        //std::vector<char> datablock(BUFFER_SIZE);
        //data = &datablock[0];

        result = ov_read(&oggStream, data, BUFFER_SIZE, 0, 2, 1, &section);

		if(result > 0){

			mysize += result;
			outdata.resize(mysize);
			memcpy(&outdata[mysize-result], data, result  );
		} else if(result < 0){
			switch(result){
				case OV_HOLE:
					printf("Interruption in the data.%d\n", result);
					printf("one of: garbage between pages, loss of sync followed by recapture, or a corrupt page\n");
					break;
				case OV_EBADLINK:
					printf("an invalid stream section was supplied to libvorbisfile, or the requested link is corrupt.\n");
					break;
				case OV_EINVAL:
					printf("the initial file headers can't be read or are corrupt, or the initial open call for vf failed.\n");
					break;
			}

		} else if(mysize == 0){
			printf("Data not read.\n");
		}
	}
	//free(dyn_data);
	return outdata;
}

void AudioPlayerAL::LoadSamplesBackground()
{
    std::vector<uint8_t> entire_file;
    {
        FILE* mysoundsfile = fopen("allsounds.dat", "rb");
        if (!mysoundsfile) {
            std::cerr << "Error: Could not open allsounds.dat. Sample loading skipped." << std::endl;
            {
                std::lock_guard<std::mutex> lock(m_loadingMutex);
                m_samplesLoaded = true;
            }
            m_loadingCV.notify_all();
            return;
        }
        fseek(mysoundsfile, 0, SEEK_END);
        size_t total_size = ftell(mysoundsfile);
        fseek(mysoundsfile, 0, SEEK_SET);
        entire_file.resize(total_size);
        if (fread(entire_file.data(), 1, total_size, mysoundsfile) != total_size) {
            std::cerr << "Error reading allsounds.dat" << std::endl;
            fclose(mysoundsfile);
            {
                std::lock_guard<std::mutex> lock(m_loadingMutex);
                m_samplesLoaded = true;
            }
            m_loadingCV.notify_all();
            return;
        }
        fclose(mysoundsfile);
    }

    size_t offset = 0;
    for (size_t i = 0; i < allsamples.size(); i++)
    {
        for (size_t j = 0; j < allsamples[i].size(); j++)
        {
            if (allsamples[i][j].size() > 0)
            {
                if (offset + sizeof(uint32_t) > entire_file.size()) break;
                uint32_t sample_filesize;
                memcpy(&sample_filesize, &entire_file[offset], sizeof(uint32_t));
                offset += sizeof(uint32_t);

                if (offset + sample_filesize > entire_file.size()) break;
                allsamples[i][j] = snd_load_file(&entire_file[offset], sample_filesize);

                int16_t * idata = ((int16_t *) &allsamples[i][j][0]);

                size_t realsize11 = sample_filesize/8;

              //  allsamples_11[i][j].resize(allsamples[i][j].size()/2);
             //   for ( int64_t k = 0; k < allsamples[i][j].size()/2; k++ )
              //  {    
              //      allsamples_11[i][j][k] = idata[k] / 32768.0f;
              //  }

static const float FIR_COEFFS[15] = {
    -0.0026f, -0.0067f, -0.0114f,  0.0000f,  0.0483f,  0.1319f,  0.2153f,  0.2505f,
     0.2153f,  0.1319f,  0.0483f,  0.0000f, -0.0114f, -0.0067f, -0.0026f
};
                allsamples_11[i][j].resize(realsize11*sizeof(int16_t));
                int16_t* idata11 = reinterpret_cast<int16_t*>(allsamples_11[i][j].data());

                // Polyphase Decimation Loop
                for (size_t k = 0; k < realsize11; k++)
                {    
                   size_t target_idx = k * 4;
                   float filtered_sample = 0.0f;

                    // Check boundaries or ensure padding exists on total_samples_44k
                    if (target_idx + 15 < sample_filesize/2) {
                        for (int tap = 0; tap < 15; tap++) {
                           filtered_sample += static_cast<float>(idata[target_idx + tap]) * FIR_COEFFS[tap];
                    }
                    } else {
                    filtered_sample = static_cast<float>(idata[target_idx]); // Fallback boundary constraint
                    }

                   idata11[k] = static_cast<int16_t>(filtered_sample);
                } 

                offset += sample_filesize;
            }
        }
    }



    {
        std::lock_guard<std::mutex> lock(m_loadingMutex);
        m_samplesLoaded = true;
    }
    m_loadingCV.notify_all();
}

void AudioPlayerAL::SendToneTuples(std::vector<std::vector<ToneTuple>> toneTuples)
{
    ABCInput * newabc = new ABCInput();
    newabc->LoadToneTuples(toneTuples);
    if (myabc == NULL)
    {
        myabc = newabc;
    }
    else
    {
        delete(myabc);
        myabc = newabc;
    }

    m_Nabctracks = newabc->Nabctracks();
    m_mutes.resize(m_Nabctracks, false);
    m_durationseconds = newabc->GetDuration();

    if ((m_Nabctracks>0) && (newabc->GetID(0) <= 0))
    {
        // this ABC didn't have Track ID info so we're setting this
        for (int i = 0; i < static_cast<int>(m_Nabctracks); i++) myabc->SetID(i, i+1);
    }
}

void AudioPlayerAL::SendABC(std::stringstream * abctext)
{
   // WaitForSamples();
    ABCInput * newabc = new ABCInput();

    newabc->LoadABC(abctext);

    m_Nabctracks = newabc->Nabctracks();
    m_originalpartnumbers = newabc->GetOriginalPartNumbers();

   // m_instrumentnumber.resize(m_Nabctracks);
    m_mutes.resize(m_Nabctracks, false);


    m_durationseconds = newabc->GetDuration();
    //  m_ABCTonesvector = newabc->m_ABCTonesvector;



    if (myabc == NULL)
    {
        myabc = newabc;
    }
    else
    {
        delete(myabc);
        myabc = newabc;
    }

    if ((m_Nabctracks>0) && (newabc->GetID(0) == -1))
    {
        // this ABC didn't have Track ID info so we're setting this
        for (int i = 0; i < static_cast<int>(m_Nabctracks); i++) myabc->SetID(i, i+1);
    }

}


#endif // AUDIOPLAYERAL_H_INCLUDED
