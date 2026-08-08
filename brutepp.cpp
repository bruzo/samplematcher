// brutepp.cpp  —  SampleMatcher GUI
// GLFW3 + ImGui (OpenGL3) front-end around the simulated-annealing sample matcher.


// TODO: Delta-fingerprint instead of full fingerprint ... 

// Split ABC calculation into per track calculation instead of all at once
// only recalculate ABC tracks that were changed in the current move

// split ABC import into per track import in the player / allow partial abc import while rest stays the same
// delta wav computation: when partial track is updated - for small number of changed tones : substract the old ones + add the new ones 
// Fingerprint: only recalculate the fingerprint in areas that changed in the wav  ( info must come from the wav generator )
// Fingerprintmatch: keep squared value: for delta frame: substract rms addition from old frames, add new ones

#include "stdio.h"
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <fstream>
#include <cstdint>
#include <list>
#include <complex>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <random>
#include <thread>
#include <mutex>
#include <atomic>
#include <omp.h>
#include <chrono>
#include <limits>

// ImGui + GLFW + OpenGL3
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>   // includes GL headers on Linux

#ifdef win32
#include "windows.h"
#include "commctrl.h"
#endif

#include "include/brute.h"
#include "include/brutedefinitions.h"
#include "include/audioplayerAL.h"
#include "midifile/include/MidiFile.h"
#define POCKETFFT_NO_MULTITHREADING
#include "pocketfft/pocketfft_hdronly.h"

#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif

// ---------------------------------------------------------------------------
const int precision=44;
const uint32_t twowindows = 2048;
const uint32_t onewindow  = 1024;


/*
const int precision=11;
const uint32_t twowindows = 512;
const uint32_t onewindow = 256;
*/

// ---------------------------------------------------------------------------
// Cooling parameters
// ---------------------------------------------------------------------------
const double coolingrate = 0.99995;
const uint64_t reheat_intervall = 2000000;

// ---------------------------------------------------------------------------
// Hann window
// ---------------------------------------------------------------------------
std::vector<float> createHannWindow(size_t size) {
    std::vector<float> hann(size);
    if (size == 0) return hann;
    const float pi = std::acos(-1.0f);
    for (size_t i = 0; i < size; ++i)
        hann[i] = 0.5f * (1.0f - std::cos(2.0f * pi * i / size));
    return hann;
}
std::vector<float> hann;
std::vector<float> hann_11;

// ---------------------------------------------------------------------------
// Statistics (only collected when --stats is passed on the command line)
// ---------------------------------------------------------------------------
struct StatRecord {
    int    moveType;      // -1=noop, 0=delete, 1=add, 2=modify, 3=track-switch
    double energyDiff;   // candScore - currentScore  (negative = improvement)
    bool   accepted;
    bool   newBest;
};

bool                    g_statsEnabled = false;
std::vector<StatRecord> g_statsRecords; // written by worker, read by main after join

// Human-readable label for a move type
static const char* moveTypeName(int action) {
    switch (action) {
        case -1: return "noop";
        case  0: return "delete";
        case  1: return "add";
        case  2: return "modify";
        case  3: return "track-switch";
        default: return "unknown";
    }
}

// Write accumulated stats to a TSV file
static void saveStats(const std::string& path) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "[stats] Could not open " << path << " for writing\n";
        return;
    }
    f << "move_type\tenergy_diff\taccepted\tnew_best\n";
    for (const auto& r : g_statsRecords) {
        f << moveTypeName(r.moveType) << "\t"
          << r.energyDiff            << "\t"
          << (r.accepted ? 1 : 0)   << "\t"
          << (r.newBest  ? 1 : 0)   << "\n";
    }
    std::cout << "[stats] Saved " << g_statsRecords.size() << " records to " << path << "\n";
}

// ---------------------------------------------------------------------------
// A-weighting ( for human perception )
// ---------------------------------------------------------------------------
std::vector<float> createAWeighting(size_t size)
{
    std::vector<float> a_weighting(size);
    for (size_t i=0; i<101; i++)
    {
        double freq = i* 44100.0 / 1024;
        a_weighting[i] = 12194*12194. *freq*freq*freq*freq / (( freq*freq+20.6*20.6)*(freq*freq+12194*12194) * 
                              std::sqrt(  (freq*freq+107.7*107.7) * (freq*freq+737.9*737.9) ));
    }
    return a_weighting;
}
std::vector<float> a_weighting;


// ---------------------------------------------------------------------------
// WAV loader  (44.1 kHz, 16-bit signed, mono assumed)
// ---------------------------------------------------------------------------
std::vector<float> LoadTargetSample(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << "\n";
        return {};
    }
    file.seekg(40, std::ios::beg);
    uint32_t dataSize;
    file.read(reinterpret_cast<char*>(&dataSize), 4);

    std::vector<int16_t> rawBuffer(dataSize / 2);
    file.read(reinterpret_cast<char*>(rawBuffer.data()), dataSize);
    file.close();

    std::vector<float> out;
    out.reserve(rawBuffer.size());
    for (int16_t s : rawBuffer) out.push_back(s / 32768.0f);

    // zero-pad to next multiple of 512
    size_t padding = (out.size() + 512) % 512;
    for (size_t i = 0; i < padding; i++) out.push_back(0.f);

    std::cout << "Loaded " << out.size() << " samples from " << filename << "\n";
    return out;
}

// Types Tone and ToneList are now defined in include/brute.h


// ---------------------------------------------------------------------------
// ToneList file I/O helpers
// ---------------------------------------------------------------------------
ToneList LoadToneListFromFile(const std::string& path)
{
    ToneList tl;
    std::ifstream f(path);
    if (!f.is_open()) return tl;

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            int t, instr, poly = 1;
            int scanned = std::sscanf(line.c_str(), "# %d %d %d", &t, &instr, &poly);
            if (scanned >= 2) {
                if (t >= (int)tl.tones.size()) {
                    tl.tones.resize(t + 1);
                    tl.midiinstruments.resize(t + 1, 0);
                    tl.polyphony.resize(t + 1, 1);
                }
                tl.midiinstruments[t] = (uint8_t)instr;
                tl.polyphony[t] = (uint8_t)poly;
            }
        } else {
            int t, p;
            uint64_t s, d, v;
            if (std::sscanf(line.c_str(), "%d %llu %llu %d %d", &t, &s, &d, &p, &v) == 5) {
                if (t >= (int)tl.tones.size()) {
                    tl.tones.resize(t + 1);
                    tl.midiinstruments.resize(t + 1, 0);
                    tl.polyphony.resize(t + 1, 1);
                }
                tl.tones[t].push_back({s, d, (uint8_t)v, (uint8_t)p});
            }
        }
    }
    tl.notracks = (uint8_t)tl.tones.size();
    tl.optimizeEnabled.resize(tl.tones.size(), true);
    tl.polyphony.resize(tl.tones.size(), 1);
    return tl;
}

void SaveToneListToFile(const ToneList& tl, const std::string& path)
{
    std::ofstream f(path);
    if (!f.is_open()) return;
    for (size_t t = 0; t < tl.tones.size(); ++t) {
        uint8_t instr = (t < tl.midiinstruments.size()) ? tl.midiinstruments[t] : 0;
        uint8_t poly = (t < tl.polyphony.size()) ? tl.polyphony[t] : 1;
        f << "# " << t << " " << (int)instr << " " << (int)poly << "\n";
        for (const auto& tone : tl.tones[t])
            f << t << " " << tone.start << " " << tone.duration << " " << (int)tone.pitch<< " " << (int)tone.velocity <<"\n";
    }
}

void WriteToneTuplesToFile(const std::string& filename, const std::vector<std::vector<ToneTuple>>& tuples)
    {
        std::ofstream f(filename);
        if (!f.is_open()) return;

        for (size_t track = 0; track < tuples.size(); ++track) {
            f << "Track " << track << ":\n";
            for (size_t i = 0; i < tuples[track].size(); ++i) {
                f << "  " << std::get<0>(tuples[track][i]) << ", " 
                << std::get<1>(tuples[track][i]) << ", " 
                << std::get<2>(tuples[track][i]) << ", " 
                << std::get<3>(tuples[track][i]) << ", "
                << std::get<4>(tuples[track][i]) << ", "
                << std::get<5>(tuples[track][i]) << "\n";
            }
        }
        f.close();
    }

// ---------------------------------------------------------------------------
// Mutation helpers
// ---------------------------------------------------------------------------
std::uniform_int_distribution<size_t> velocity(0, 127);
std::uniform_int_distribution<size_t> pitch(0, 36);
std::uniform_int_distribution<size_t> duration(1, 16000);

struct ToneMutation {
    int    action = -1;   // -1=noop, 0=delete, 1=add, 2=modify, 3=track-switch
    size_t trackId = 0;
    size_t targetTrackId = 0;  // used by action==3 (track-switch): destination track
    size_t deletedToneIndex = 0;
    Tone   deletedTone;
};

ToneMutation mutate(ToneList& tl, std::mt19937& gen, uint64_t thissize, uint64_t tonelimit)
{
    ToneMutation mut;
    if (tl.tones.empty()) return mut;

    std::vector<size_t> enabledTracks;
    for (size_t t = 1; t < tl.tones.size(); ++t) {
        bool opt = true;
        if (t < tl.optimizeEnabled.size()) {
            opt = tl.optimizeEnabled[t];
        }
        if (opt) {
            enabledTracks.push_back(t);
        }
    }

    if (enabledTracks.empty()) return mut;

    std::uniform_int_distribution<size_t> uni_track(0, enabledTracks.size() - 1);
    std::uniform_int_distribution<size_t> uni_action(0, 16);

    // count tones in tonelist
    uint64_t n_tones = 0;
    for (size_t j = 0; j < tl.tones.size(); j++)
       n_tones += tl.tones[j].size();

    size_t action  = uni_action(gen);
    size_t trackId = enabledTracks[uni_track(gen)];

    if (n_tones == 0) action = 1;

    if (action == 0) { // Delete a Tone
        if (tl.tones[trackId].size() > 0) {
            std::uniform_int_distribution<size_t> uni_tone(0, tl.tones[trackId].size() - 1);
            size_t mytone = uni_tone(gen);
            mut.action           = 0;
            mut.trackId          = trackId;
            mut.deletedToneIndex = mytone;
            mut.deletedTone      = tl.tones[trackId][mytone];
            std::swap(tl.tones[trackId][mytone], tl.tones[trackId].back());
            tl.tones[trackId].pop_back();
        }
    } else if ((action == 1)&& (n_tones < tonelimit) ){  // Add a tone
        Tone newtone;
        std::uniform_int_distribution<uint64_t> start(0, 1024 * static_cast<uint64_t>(thissize));
        newtone.duration = duration(gen);
        newtone.velocity = velocity(gen);
        newtone.pitch    = pitch(gen);
        newtone.start    = start(gen);
        mut.action  = 1;
        mut.trackId = trackId;
        tl.tones[trackId].push_back(std::move(newtone));
    } else {  // totally change a tone
        if ((action==2)&&(tl.tones[trackId].size() > 0)) {
            std::uniform_int_distribution<size_t>   uni_tone(0, tl.tones[trackId].size() - 1);
            std::uniform_int_distribution<uint64_t> start(0, 1024 * static_cast<uint64_t>(thissize));
            size_t mytone = uni_tone(gen);
            mut.action           = 2;
            mut.trackId          = trackId;
            mut.deletedToneIndex = mytone;
            mut.deletedTone.pitch    = tl.tones[trackId][mytone].pitch;
            mut.deletedTone.velocity = tl.tones[trackId][mytone].velocity;
            mut.deletedTone.start    = tl.tones[trackId][mytone].start;
            mut.deletedTone.duration = tl.tones[trackId][mytone].duration;
            tl.tones[trackId][mytone].pitch    = pitch(gen);
            tl.tones[trackId][mytone].velocity = velocity(gen);
            tl.tones[trackId][mytone].start    = start(gen);
            tl.tones[trackId][mytone].duration = duration(gen);
        }
        if ((action==3)&&(tl.tones[trackId].size() > 0)) {  // change the pitch of a tone
            std::uniform_int_distribution<size_t>   uni_tone(0, tl.tones[trackId].size() - 1);
            std::uniform_int_distribution<uint64_t> dpitch(0, 6);
            size_t mytone = uni_tone(gen);
            mut.action           = 2;
            mut.trackId          = trackId;
            mut.deletedToneIndex = mytone;
            mut.deletedTone.pitch    = tl.tones[trackId][mytone].pitch;
            mut.deletedTone.velocity = tl.tones[trackId][mytone].velocity;
            mut.deletedTone.start    = tl.tones[trackId][mytone].start;
            mut.deletedTone.duration = tl.tones[trackId][mytone].duration;

            uint64_t mypitchdiff = dpitch(gen);
            int8_t pitchjump = 0;
            if (mypitchdiff==0) pitchjump = -12;
            if (mypitchdiff==1) pitchjump = -7;
            if (mypitchdiff==2) pitchjump = -1;
            if (mypitchdiff==3) pitchjump = 1;
            if (mypitchdiff==4) pitchjump = 1;
            if (mypitchdiff==5) pitchjump = 7;
            if (mypitchdiff==6) pitchjump = 12;

            pitchjump = pitchjump + tl.tones[trackId][mytone].pitch;
            if (pitchjump < 0) pitchjump = 0;
            if (pitchjump > 36) pitchjump = 36;
            tl.tones[trackId][mytone].pitch  = pitchjump;
        }
        if ((action>=4)&&(action <= 8)&&(tl.tones[trackId].size() > 0)) {   // change start of a tone
            std::uniform_int_distribution<size_t>   uni_tone(0, tl.tones[trackId].size() - 1);
            std::uniform_int_distribution<uint64_t> dpos(-1024, 1024);
            size_t mytone = uni_tone(gen);

            mut.action           = 2;
            mut.trackId          = trackId;
            mut.deletedToneIndex = mytone;
            mut.deletedTone.pitch    = tl.tones[trackId][mytone].pitch;
            mut.deletedTone.velocity = tl.tones[trackId][mytone].velocity;
            mut.deletedTone.start    = tl.tones[trackId][mytone].start;
            mut.deletedTone.duration = tl.tones[trackId][mytone].duration;

            int64_t myposdiff = dpos(gen);
           // if (myposdiff < 0) myposdiff = myposdiff - 512;
           // if (myposdiff > 0) myposdiff = myposdiff + 512;
            myposdiff = myposdiff + tl.tones[trackId][mytone].start;
            if (myposdiff < 0) myposdiff = 0;
            tl.tones[trackId][mytone].start  = myposdiff;
        }
        if ((action>=9)&&(action<=13)&&(tl.tones[trackId].size() > 0)) {   // change duration of a tone
            std::uniform_int_distribution<size_t>   uni_tone(0, tl.tones[trackId].size() - 1);
            std::uniform_int_distribution<uint64_t> dpos(-1024, 1024);
            size_t mytone = uni_tone(gen);

            mut.action           = 2;
            mut.trackId          = trackId;
            mut.deletedToneIndex = mytone;
            mut.deletedTone.pitch    = tl.tones[trackId][mytone].pitch;
            mut.deletedTone.velocity = tl.tones[trackId][mytone].velocity;
            mut.deletedTone.start    = tl.tones[trackId][mytone].start;
            mut.deletedTone.duration = tl.tones[trackId][mytone].duration;

            int64_t myposdiff = dpos(gen);
          //  if (myposdiff < 0) myposdiff = myposdiff - 512;
          //  if (myposdiff > 0) myposdiff = myposdiff + 512;
            myposdiff = myposdiff + tl.tones[trackId][mytone].duration;
            if (myposdiff < 256) myposdiff = 256;
            tl.tones[trackId][mytone].duration  = myposdiff;
        }
        if ((action>13)&&(action<=15)&&(tl.tones[trackId].size() > 0)) {  // change velocity of a tone 

            std::uniform_int_distribution<size_t>   uni_tone(0, tl.tones[trackId].size() - 1);
            size_t mytone = uni_tone(gen);

            mut.action           = 2;
            mut.trackId          = trackId;
            mut.deletedToneIndex = mytone;
            mut.deletedTone.pitch    = tl.tones[trackId][mytone].pitch;
            mut.deletedTone.velocity = tl.tones[trackId][mytone].velocity;
            mut.deletedTone.start    = tl.tones[trackId][mytone].start;
            mut.deletedTone.duration = tl.tones[trackId][mytone].duration;

            tl.tones[trackId][mytone].velocity  = velocity(gen);
        }
        if ((action == 16) && (tl.tones[trackId].size() > 0) && (enabledTracks.size() > 1)) {  // move tone to another track
            // Pick a different destination track.
            std::vector<size_t> otherTracks;
            for (size_t t : enabledTracks)
                if (t != trackId) otherTracks.push_back(t);
            std::uniform_int_distribution<size_t> uni_other(0, otherTracks.size() - 1);
            size_t destTrack = otherTracks[uni_other(gen)];

            std::uniform_int_distribution<size_t> uni_tone(0, tl.tones[trackId].size() - 1);
            size_t mytone = uni_tone(gen);

            mut.action           = 3;
            mut.trackId          = trackId;
            mut.targetTrackId    = destTrack;
            mut.deletedToneIndex = mytone;
            mut.deletedTone      = tl.tones[trackId][mytone];

            // Move: append to destination, then remove from source (swap-and-pop).
            tl.tones[destTrack].push_back(tl.tones[trackId][mytone]);
            std::swap(tl.tones[trackId][mytone], tl.tones[trackId].back());
            tl.tones[trackId].pop_back();
        }
    }
    return mut;
}

void revertMutation(ToneList& tl, const ToneMutation& mut)
{
    if (mut.action == 0) {
        tl.tones[mut.trackId].push_back(mut.deletedTone);
        std::swap(tl.tones[mut.trackId][mut.deletedToneIndex], tl.tones[mut.trackId].back());
    } else if (mut.action == 1) {
        if (!tl.tones[mut.trackId].empty())
            tl.tones[mut.trackId].pop_back();
    } else if (mut.action == 2) {
        tl.tones[mut.trackId][mut.deletedToneIndex].pitch    = mut.deletedTone.pitch;
        tl.tones[mut.trackId][mut.deletedToneIndex].velocity = mut.deletedTone.velocity;
        tl.tones[mut.trackId][mut.deletedToneIndex].start    = mut.deletedTone.start;
        tl.tones[mut.trackId][mut.deletedToneIndex].duration = mut.deletedTone.duration;
    } else if (mut.action == 3) {
        // Undo track-switch: remove the tone from destTrack (it was appended last)
        // and reinsert it at its original index in trackId.
        if (!tl.tones[mut.targetTrackId].empty())
            tl.tones[mut.targetTrackId].pop_back();
        tl.tones[mut.trackId].push_back(mut.deletedTone);
        std::swap(tl.tones[mut.trackId][mut.deletedToneIndex], tl.tones[mut.trackId].back());
    }
}

// ---------------------------------------------------------------------------
// Sweeping Mode Helpers
// ---------------------------------------------------------------------------

// Extract a sub-ToneList for a sliding-window pass over the full ToneList.
//
// The "capture zone" = [captureStart, captureEnd) where:
//   captureStart = max(0, windowStart - ghostSize), aligned DOWN to the 1024-sample FFT grid
//   captureEnd   = windowEnd + ghostSize
//
// All tone.start values are shifted left by outOffset (= captureStart) so the
// local audio renderer always sees offsets starting near 0.
//
// The "active window" in local (shifted) coordinates is:
//   [windowStart - outOffset,  windowEnd - outOffset)
// Tones inside the ghost area but outside the active window are included for
// rendering context but skipped by windowedMutate() based on their start position.
ToneList extractWindowToneList(
    const ToneList& full,
    uint64_t windowStart,
    uint64_t windowEnd,
    uint64_t ghostSize,
    uint64_t& outOffset)
{
    // Capture range: align captureStart down to 1024-sample FFT boundary
    uint64_t captureStart = (windowStart > ghostSize) ? windowStart - ghostSize : 0;
    captureStart = (captureStart / 1024) * 1024;
    uint64_t captureEnd = windowEnd + ghostSize;
    outOffset = captureStart;

    ToneList result;
    result.notracks        = full.notracks;
    result.tones.resize(full.tones.size());
    result.midiinstruments = full.midiinstruments;
    result.optimizeEnabled = full.optimizeEnabled;
    result.polyphony       = full.polyphony;

    for (size_t t = 0; t < full.tones.size(); ++t) {
        for (const auto& tone : full.tones[t]) {
            uint64_t toneEnd = tone.start + tone.duration;
            // Include tone if it overlaps [captureStart, captureEnd)
            if (toneEnd > captureStart && tone.start < captureEnd) {
                Tone shifted  = tone;
                shifted.start = (tone.start >= outOffset) ? tone.start - outOffset : 0;
                result.tones[t].push_back(shifted);
            }
        }
    }
    return result;
}

// Merge a locally-optimised window ToneList back into the full ToneList.
// Replaces ONLY tones whose original start falls in [windowStart, windowEnd).
// windowOffset is the shift applied during extraction (= captureStart).
void mergeWindowToneList(
    ToneList& full,
    const ToneList& windowResult,
    uint64_t windowStart,
    uint64_t windowEnd,
    uint64_t windowOffset)
{
    uint64_t localWindowStart = windowStart - windowOffset;
    uint64_t localWindowEnd   = windowEnd   - windowOffset;

    for (size_t t = 0; t < full.tones.size() && t < windowResult.tones.size(); ++t) {
        // Remove full-list tones that fall in [windowStart, windowEnd)
        auto& v = full.tones[t];
        v.erase(
            std::remove_if(v.begin(), v.end(),
                [windowStart, windowEnd](const Tone& tone) {
                    return tone.start >= windowStart && tone.start < windowEnd;
                }),
            v.end());

        // Re-insert the optimised tones (only active-window tones, with offset restored)
        for (const auto& tone : windowResult.tones[t]) {
            if (tone.start >= localWindowStart && tone.start < localWindowEnd) {
                Tone restored  = tone;
                restored.start += windowOffset;
                full.tones[t].push_back(restored);
            }
        }
    }
}

// Window-aware mutation: like mutate() but only picks / places tones whose
// start falls in [localWindowStart, localWindowEnd) (local / shifted coords).
// Ghost tones outside that range are present in tl for rendering but are
// never selected here.
ToneMutation windowedMutate(
    ToneList& tl,
    std::mt19937& gen,
    uint64_t localWindowStart,
    uint64_t localWindowEnd,
    uint64_t tonelimit)
{
    ToneMutation mut;
    if (tl.tones.empty()) return mut;

    // Collect enabled tracks
    std::vector<size_t> enabledTracks;
    for (size_t t = 1; t < tl.tones.size(); ++t) {
        bool opt = (t < tl.optimizeEnabled.size()) ? tl.optimizeEnabled[t] : true;
        if (opt) enabledTracks.push_back(t);
    }
    if (enabledTracks.empty()) return mut;

    // Build per-track index lists restricted to the active window
    std::vector<std::vector<size_t>> editableIdx(tl.tones.size());
    uint64_t n_editable = 0;
    for (size_t t : enabledTracks) {
        for (size_t i = 0; i < tl.tones[t].size(); ++i) {
            uint64_t s = tl.tones[t][i].start;
            if (s >= localWindowStart && s < localWindowEnd) {
                editableIdx[t].push_back(i);
                ++n_editable;
            }
        }
    }

    // Total tones (for tonelimit enforcement)
    uint64_t n_tones = 0;
    for (size_t j = 0; j < tl.tones.size(); j++)
        n_tones += tl.tones[j].size();

    std::uniform_int_distribution<size_t> uni_track(0, enabledTracks.size() - 1);
    std::uniform_int_distribution<size_t> uni_action(0, 16);

    size_t action  = uni_action(gen);
    size_t trackId = enabledTracks[uni_track(gen)];

    // If no editable tones exist, the only sensible action is to add one
    if (n_editable == 0) {
        if (n_tones >= tonelimit) return mut; // hard limit reached, noop
        action = 1;
    }

    uint64_t winSize = (localWindowEnd > localWindowStart) ? localWindowEnd - localWindowStart : 1024;

    if (action == 0) { // Delete a tone from the active window
        if (!editableIdx[trackId].empty()) {
            std::uniform_int_distribution<size_t> uni_e(0, editableIdx[trackId].size() - 1);
            size_t eIdx = editableIdx[trackId][uni_e(gen)];
            mut.action           = 0;
            mut.trackId          = trackId;
            mut.deletedToneIndex = eIdx;
            mut.deletedTone      = tl.tones[trackId][eIdx];
            std::swap(tl.tones[trackId][eIdx], tl.tones[trackId].back());
            tl.tones[trackId].pop_back();
        }

    } else if ((action == 1) && (n_tones < tonelimit)) { // Add a tone inside the active window
        Tone newtone;
        uint64_t slots = winSize / 1024;
        if (slots == 0) slots = 1;
        std::uniform_int_distribution<uint64_t> startDist(0, slots - 1);
        newtone.start    = localWindowStart + startDist(gen) * 1024;
        newtone.duration = duration(gen);
        newtone.velocity = (uint8_t)velocity(gen);
        newtone.pitch    = (uint8_t)pitch(gen);
        mut.action  = 1;
        mut.trackId = trackId;
        tl.tones[trackId].push_back(std::move(newtone));

    } else if (action >= 2 && !editableIdx[trackId].empty()) {
        std::uniform_int_distribution<size_t> uni_e(0, editableIdx[trackId].size() - 1);
        size_t eIdx = editableIdx[trackId][uni_e(gen)];

        mut.action           = 2;
        mut.trackId          = trackId;
        mut.deletedToneIndex = eIdx;
        mut.deletedTone.pitch    = tl.tones[trackId][eIdx].pitch;
        mut.deletedTone.velocity = tl.tones[trackId][eIdx].velocity;
        mut.deletedTone.start    = tl.tones[trackId][eIdx].start;
        mut.deletedTone.duration = tl.tones[trackId][eIdx].duration;

        if (action == 2) { // Fully replace the tone
            uint64_t slots = winSize / 1024;
            if (slots == 0) slots = 1;
            std::uniform_int_distribution<uint64_t> startDist(0, slots - 1);
            tl.tones[trackId][eIdx].start    = localWindowStart + startDist(gen) * 1024;
            tl.tones[trackId][eIdx].pitch    = (uint8_t)pitch(gen);
            tl.tones[trackId][eIdx].velocity = (uint8_t)velocity(gen);
            tl.tones[trackId][eIdx].duration = duration(gen);

        } else if (action == 3) { // Nudge pitch
            std::uniform_int_distribution<uint64_t> dpitch(0, 6);
            uint64_t mypitchdiff = dpitch(gen);
            int8_t pitchjump = 0;
            if      (mypitchdiff == 0) pitchjump = -12;
            else if (mypitchdiff == 1) pitchjump = -7;
            else if (mypitchdiff == 2) pitchjump = -1;
            else if (mypitchdiff == 3) pitchjump =  1;
            else if (mypitchdiff == 4) pitchjump =  1;
            else if (mypitchdiff == 5) pitchjump =  7;
            else if (mypitchdiff == 6) pitchjump =  12;
            pitchjump += (int8_t)tl.tones[trackId][eIdx].pitch;
            if (pitchjump < 0)  pitchjump = 0;
            if (pitchjump > 36) pitchjump = 36;
            tl.tones[trackId][eIdx].pitch = (uint8_t)pitchjump;

        } else if (action >= 4 && action <= 8) { // Nudge start (clamped to active window)
            std::uniform_int_distribution<int64_t> dpos(-1024, 1024);
            int64_t newstart = (int64_t)tl.tones[trackId][eIdx].start + dpos(gen);
            if (newstart < (int64_t)localWindowStart) newstart = (int64_t)localWindowStart;
            if (newstart >= (int64_t)localWindowEnd)  newstart = (int64_t)(localWindowEnd - 1024);
            if (newstart < (int64_t)localWindowStart) newstart = (int64_t)localWindowStart;
            tl.tones[trackId][eIdx].start = (uint64_t)newstart;

        } else if (action >= 9 && action <= 13) { // Nudge duration
            std::uniform_int_distribution<int64_t> dpos(-1024, 1024);
            int64_t newdur = (int64_t)tl.tones[trackId][eIdx].duration + dpos(gen);
            if (newdur < 256) newdur = 256;
            tl.tones[trackId][eIdx].duration = (uint64_t)newdur;

        } else if (action >= 14 && action <= 15) { // Change velocity
            tl.tones[trackId][eIdx].velocity = (uint8_t)velocity(gen);

        } else if (action == 16 && enabledTracks.size() > 1) { // Move tone to another track
            std::vector<size_t> otherTracks;
            for (size_t t : enabledTracks)
                if (t != trackId) otherTracks.push_back(t);
            std::uniform_int_distribution<size_t> uni_other(0, otherTracks.size() - 1);
            size_t destTrack = otherTracks[uni_other(gen)];

            mut.action        = 3;
            mut.targetTrackId = destTrack;

            tl.tones[destTrack].push_back(tl.tones[trackId][eIdx]);
            std::swap(tl.tones[trackId][eIdx], tl.tones[trackId].back());
            tl.tones[trackId].pop_back();
        }
    }
    return mut;
}

// ---------------------------------------------------------------------------
// MIDI generation
// ---------------------------------------------------------------------------
smf::MidiFile createMidiFromToneList(const ToneList& toneList, int ticksPerQuarterNote)
{
    smf::MidiFile midiFile;
    midiFile.setTicksPerQuarterNote(ticksPerQuarterNote);

    size_t trackCount = toneList.tones.size();
    while (midiFile.getTrackCount() < (int)trackCount)
        midiFile.addTrack();

    for (size_t trackIdx = 0; trackIdx < trackCount; ++trackIdx) {
        if (trackIdx < toneList.midiinstruments.size())
            midiFile.addTimbre((int)trackIdx, 0, 0, toneList.midiinstruments[trackIdx]);
        midiFile.addController((int)trackIdx, 0, 0, 7, 100); // CC7: channel volume = 100

        double samplesPerQN   = 44100.0 / 2.0;
        double ticksPerSample = ticksPerQuarterNote / samplesPerQN;

        for (const auto& tone : toneList.tones[trackIdx]) {
            int startTick = static_cast<int>(tone.start * ticksPerSample);
            int endTick   = static_cast<int>((tone.start + tone.duration) * ticksPerSample);
            if (startTick < 0)        startTick = 0;
            if (endTick <= startTick) endTick = startTick + 1;
            midiFile.addNoteOn ((int)trackIdx, startTick, 0, tone.pitch, tone.velocity);
            midiFile.addNoteOff((int)trackIdx, endTick,   0, tone.pitch, 0);
        }
    }
    midiFile.sortTracks();
    return midiFile;
}

// ---------------------------------------------------------------------------
// FFT Fingerprint
// ---------------------------------------------------------------------------
struct FFTBin { float real; };
const int BINS_TO_KEEP = 101;
using FingerprintFrame = std::array<FFTBin, BINS_TO_KEEP>;



std::vector<FingerprintFrame> FingerPrint(const std::vector<float>& targetSample)
{
    const int FFT_SIZE = 1024;
    const int R2C_OUT_SIZE = (FFT_SIZE / 2) + 1; 

    std::vector<FingerprintFrame> fp;
    if (targetSample.size() >= (size_t)FFT_SIZE)
        fp.reserve(2 * targetSample.size() / FFT_SIZE);

    static const pocketfft::shape_t  shape_in  { (size_t)FFT_SIZE };
    static const pocketfft::shape_t  axes      { 0 };
    
    static const pocketfft::stride_t stride_in  { (ptrdiff_t)sizeof(float) };
    static const pocketfft::stride_t stride_out { (ptrdiff_t)sizeof(std::complex<float>) };

    // Thread-local scratch buffers
    static thread_local std::vector<float> in(FFT_SIZE);
    static thread_local std::vector<std::complex<float>> out(R2C_OUT_SIZE);

    for (size_t i = 0; i + FFT_SIZE <= targetSample.size(); i += FFT_SIZE/2) {
        
        // 1. Vectorized Windowing Loop
        // Hint alignment and inform the compiler there is no pointer aliasing
        const float* __restrict src = &targetSample[i];
        float* __restrict dest = in.data();
        
        #pragma omp simd
        for (int k = 0; k < FFT_SIZE; ++k) {
            dest[k] = src[k] * hann[k];
        }

        // Execute FFT
        pocketfft::r2c(shape_in, stride_in, stride_out, axes, pocketfft::FORWARD,
                       in.data(), out.data(), 1.0f);

        FingerprintFrame frame;
        
        // 2. Vectorized Magnitude Loop
        // Cast to raw float array to avoid std::complex layout tracking penalties
        const float* __restrict out_raw = reinterpret_cast<const float*>(out.data());
        float* __restrict frame_raw = reinterpret_cast<float*>(frame.data());
        const float* __restrict weight = &a_weighting[0];

        #pragma omp simd
        for (int k = 0; k < BINS_TO_KEEP; ++k) {
            float real = out_raw[2 * k];
            float imag = out_raw[2 * k + 1];
            float mag = std::sqrt(real * real + imag * imag);
            frame_raw[k] = mag * weight[k]; // to be inline with the 256 point FFTs
        }
        
        fp.push_back(frame);
    }
    return fp;
}

std::vector<FingerprintFrame> FingerPrint_11(const std::vector<float>& targetSample)
{
    // 1. Updated Window and Bin Constraints
    const int FFT_SIZE = 256;                        // Was 1024
    const int R2C_OUT_SIZE = (FFT_SIZE / 2) + 1;    // Now 129 (Max possible bins at 11kHz Nyquist)

    std::vector<FingerprintFrame> fp;
    if (targetSample.size() >= (size_t)FFT_SIZE)
        fp.reserve(2 * targetSample.size() / FFT_SIZE);

    // 2. Updated PocketFFT shape geometries for 256 points
    static const pocketfft::shape_t  shape_in  { (size_t)FFT_SIZE };
    static const pocketfft::shape_t  axes      { 0 };
    
    static const pocketfft::stride_t stride_in  { (ptrdiff_t)sizeof(float) };
    static const pocketfft::stride_t stride_out { (ptrdiff_t)sizeof(std::complex<float>) };

    // Thread-local scratch buffers (automatically scales down memory allocation)
    static thread_local std::vector<float> in(FFT_SIZE);
    static thread_local std::vector<std::complex<float>> out(R2C_OUT_SIZE);

    // 3. Loop step updates to FFT_SIZE/2 (128 sample hop size for 50% overlap)
    for (size_t i = 0; i + FFT_SIZE <= targetSample.size(); i += FFT_SIZE/2) {
        
        const float* __restrict src = &targetSample[i];
        float* __restrict dest = in.data();
        
        // Vectorized Windowing Loop (Ensure 'hann_256' array is provided)
        #pragma omp simd
        for (int k = 0; k < FFT_SIZE; ++k) {
            dest[k] = src[k] * hann_11[k]; 
        }

        // Execute 256-point FFT
        pocketfft::r2c(shape_in, stride_in, stride_out, axes, pocketfft::FORWARD,
                       in.data(), out.data(), 1.0f);

        FingerprintFrame frame;
        
        // Vectorized Magnitude Loop
        const float* __restrict out_raw = reinterpret_cast<const float*>(out.data());
        float* __restrict frame_raw = reinterpret_cast<float*>(frame.data());
        const float* __restrict weight = &a_weighting[0]; // Uses the exact same coefficients

        // Note: BINS_TO_KEEP remains identical if your target range is below 5.5125 kHz.
        // It must be structurally capped at a maximum value of 129 (R2C_OUT_SIZE).
        #pragma omp simd
        for (int k = 0; k < BINS_TO_KEEP; ++k) {
            float real = out_raw[2 * k];
            float imag = out_raw[2 * k + 1];
            float mag = std::sqrt(real * real + imag * imag);
            frame_raw[k] = mag * weight[k];
        }
        
        fp.push_back(frame);
    }
    return fp;
}


std::vector<FingerprintFrame> FingerPrintI(const std::vector<int32_t>& targetSample)
{
    const int FFT_SIZE = 1024;
    const int R2C_OUT_SIZE = (FFT_SIZE / 2) + 1; 

    std::vector<FingerprintFrame> fp;
    if (targetSample.size() >= (size_t)FFT_SIZE)
        fp.reserve(2 * targetSample.size() / FFT_SIZE);

    static const pocketfft::shape_t  shape_in  { (size_t)FFT_SIZE };
    static const pocketfft::shape_t  axes      { 0 };
    
    static const pocketfft::stride_t stride_in  { (ptrdiff_t)sizeof(float) };
    static const pocketfft::stride_t stride_out { (ptrdiff_t)sizeof(std::complex<float>) };

    // Thread-local scratch buffers
    static thread_local std::vector<float> in(FFT_SIZE);
    static thread_local std::vector<std::complex<float>> out(R2C_OUT_SIZE);

    // Reciprocal scale factor to bring int32 values back down to standard float range
    const float inverseScale = 1.0f / 32768.0f;

    for (size_t i = 0; i + FFT_SIZE <= targetSample.size(); i += FFT_SIZE/2) {
        
        // 1. Vectorized Conversion & Windowing Loop
        // Hint alignment and inform the compiler there is no pointer aliasing
        const int32_t* __restrict src = &targetSample[i];
        float* __restrict dest = in.data();
        const float* __restrict win = &hann[0];
        
        #pragma omp simd
        for (int k = 0; k < FFT_SIZE; ++k) {
            // Convert to float, apply inverse scale, and apply Hann windowing in one step
            dest[k] = (static_cast<float>(src[k]) * inverseScale) * win[k];
        }

        // Execute FFT
        pocketfft::r2c(shape_in, stride_in, stride_out, axes, pocketfft::FORWARD,
                       in.data(), out.data(), 1.0f);

        FingerprintFrame frame;
        
        // 2. Vectorized Magnitude Loop
        // Cast to raw float array to avoid std::complex layout tracking penalties
        const float* __restrict out_raw = reinterpret_cast<const float*>(out.data());
        float* __restrict frame_raw = reinterpret_cast<float*>(frame.data());
        const float* __restrict weight = &a_weighting[0];

        #pragma omp simd
        for (int k = 0; k < BINS_TO_KEEP; ++k) {
            float real = out_raw[2 * k];
            float imag = out_raw[2 * k + 1];
            float mag = std::sqrt(real * real + imag * imag);
            frame_raw[k] = mag * weight[k];
        }
        
        fp.push_back(frame);
    }
    return fp;
}

std::vector<FingerprintFrame> FingerPrintI_11(const std::vector<int32_t>& targetSample)
{
    // 1. Updated Window and Bin Constraints
    const int FFT_SIZE = 256;                        // Was 1024
    const int R2C_OUT_SIZE = (FFT_SIZE / 2) + 1;    // Now 129 (Max possible bins at 11kHz Nyquist)

    std::vector<FingerprintFrame> fp;
    if (targetSample.size() >= (size_t)FFT_SIZE)
        fp.reserve(2 * targetSample.size() / FFT_SIZE);

    // 2. Updated PocketFFT shape geometries for 256 points
    static const pocketfft::shape_t  shape_in  { (size_t)FFT_SIZE };
    static const pocketfft::shape_t  axes      { 0 };
    
    static const pocketfft::stride_t stride_in  { (ptrdiff_t)sizeof(float) };
    static const pocketfft::stride_t stride_out { (ptrdiff_t)sizeof(std::complex<float>) };

    // Thread-local scratch buffers (automatically scales down memory allocation)
    static thread_local std::vector<float> in(FFT_SIZE);
    static thread_local std::vector<std::complex<float>> out(R2C_OUT_SIZE);

    // Reciprocal scale factor remains unchanged
    const float inverseScale = 1.0f / 32768.0f;

    // 3. Loop step updates to FFT_SIZE/2 (128 sample hop size for 50% overlap)
    for (size_t i = 0; i + FFT_SIZE <= targetSample.size(); i += FFT_SIZE/2) {
        
        // 1. Vectorized Conversion & Windowing Loop
        const int32_t* __restrict src = &targetSample[i];
        float* __restrict dest = in.data();
        const float* __restrict win = &hann_11[0]; // Updated to your new table name
        
        #pragma omp simd
        for (int k = 0; k < FFT_SIZE; ++k) {
            // Convert to float, apply inverse scale, and apply Hann windowing in one step
            dest[k] = (static_cast<float>(src[k]) * inverseScale) * win[k];
        }

        // Execute 256-point FFT
        pocketfft::r2c(shape_in, stride_in, stride_out, axes, pocketfft::FORWARD,
                       in.data(), out.data(), 1.0f);

        FingerprintFrame frame;
        
        // 2. Vectorized Magnitude Loop
        const float* __restrict out_raw = reinterpret_cast<const float*>(out.data());
        float* __restrict frame_raw = reinterpret_cast<float*>(frame.data());
        const float* __restrict weight = &a_weighting[0]; // Uses the exact same coefficients

        // Note: Ensure BINS_TO_KEEP is structurally capped at a maximum value of 129
        #pragma omp simd
        for (int k = 0; k < BINS_TO_KEEP; ++k) {
            float real = out_raw[2 * k];
            float imag = out_raw[2 * k + 1];
            float mag = std::sqrt(real * real + imag * imag);
            frame_raw[k] = mag * weight[k];
        }
        
        fp.push_back(frame);
    }
    return fp;
}

std::vector<FingerprintFrame> FingerPrintPartialI(
    const std::vector<int32_t>& targetSample,
    const std::vector<FingerprintFrame>& currentFP,
    size_t startFrame,
    size_t endFrame)
{
    const int FFT_SIZE = 1024;
    const int R2C_OUT_SIZE = (FFT_SIZE / 2) + 1;

    size_t totalFrames = 0;
    if (targetSample.size() >= (size_t)FFT_SIZE) {
        totalFrames = (targetSample.size() - FFT_SIZE) / (FFT_SIZE / 2) + 1;
    }

    std::vector<FingerprintFrame> fp = currentFP;
    if (fp.size() != totalFrames) {
        fp.resize(totalFrames);
    }

    startFrame = std::min(startFrame, fp.size());
    endFrame   = std::min(endFrame, fp.size());

    static const pocketfft::shape_t  shape_in  { (size_t)FFT_SIZE };
    static const pocketfft::shape_t  axes      { 0 };
    static const pocketfft::stride_t stride_in  { (ptrdiff_t)sizeof(float) };
    static const pocketfft::stride_t stride_out { (ptrdiff_t)sizeof(std::complex<float>) };

    static thread_local std::vector<float> in(FFT_SIZE);
    static thread_local std::vector<std::complex<float>> out(R2C_OUT_SIZE);

    // Reciprocal scale factor to bring int32 values back down to standard float range
    const float inverseScale = 1.0f / 32768.0f;

    for (size_t frameIdx = startFrame; frameIdx < endFrame; ++frameIdx) {
        size_t i = frameIdx * (FFT_SIZE / 2);
        if (i + FFT_SIZE > targetSample.size()) break;

        // 1. Vectorized Conversion & Windowing Loop
        // Hint alignment and inform the compiler there is no pointer aliasing
        const int32_t* __restrict src = &targetSample[i];
        float* __restrict dest = in.data();
        const float* __restrict win = &hann[0];

        #pragma omp simd
        for (int k = 0; k < FFT_SIZE; ++k) {
            // Convert to float, apply inverse scale, and apply Hann windowing in one step
            dest[k] = (static_cast<float>(src[k]) * inverseScale) * win[k];
        }

        // Execute FFT
        pocketfft::r2c(shape_in, stride_in, stride_out, axes, pocketfft::FORWARD,
                       in.data(), out.data(), 1.0f);

        FingerprintFrame frame;
        
        // 2. Vectorized Magnitude Loop
        // Cast to raw float array to avoid std::complex layout tracking penalties
        const float* __restrict out_raw = reinterpret_cast<const float*>(out.data());
        float* __restrict frame_raw = reinterpret_cast<float*>(frame.data());
        const float* __restrict weight = &a_weighting[0];

        #pragma omp simd
        for (int k = 0; k < BINS_TO_KEEP; ++k) {
            float real = out_raw[2 * k];
            float imag = out_raw[2 * k + 1];
            float mag = std::sqrt(real * real + imag * imag);
            frame_raw[k] = mag * weight[k];
        }

        fp[frameIdx] = frame;
    }
    return fp;
}

std::vector<FingerprintFrame> FingerPrintPartialI_11(
    const std::vector<int32_t>& targetSample,
    const std::vector<FingerprintFrame>& currentFP,
    size_t startFrame,
    size_t endFrame)
{
    // 1. Updated Window and Bin Constraints
    const int FFT_SIZE = 256;                        // Was 1024
    const int R2C_OUT_SIZE = (FFT_SIZE / 2) + 1;    // Now 129 (Max possible bins at 11kHz Nyquist)

    size_t totalFrames = 0;
    if (targetSample.size() >= (size_t)FFT_SIZE) {
        // Correctly calculates frame count using a 128-sample hop size
        totalFrames = (targetSample.size() - FFT_SIZE) / (FFT_SIZE / 2) + 1;
    }

    std::vector<FingerprintFrame> fp = currentFP;
    if (fp.size() != totalFrames) {
        fp.resize(totalFrames);
    }

    startFrame = std::min(startFrame, fp.size());
    endFrame   = std::min(endFrame, fp.size());

    // 2. Updated PocketFFT shape geometries for 256 points
    static const pocketfft::shape_t  shape_in  { (size_t)FFT_SIZE };
    static const pocketfft::shape_t  axes      { 0 };
    static const pocketfft::stride_t stride_in  { (ptrdiff_t)sizeof(float) };
    static const pocketfft::stride_t stride_out { (ptrdiff_t)sizeof(std::complex<float>) };

    // Thread-local scratch buffers (automatically scales down memory allocation)
    static thread_local std::vector<float> in(FFT_SIZE);
    static thread_local std::vector<std::complex<float>> out(R2C_OUT_SIZE);

    // Reciprocal scale factor remains unchanged
    const float inverseScale = 1.0f / 32768.0f;

    for (size_t frameIdx = startFrame; frameIdx < endFrame; ++frameIdx) {
        // Maps the index correctly using a 128-sample step (FFT_SIZE / 2)
        size_t i = frameIdx * (FFT_SIZE / 2);
        if (i + FFT_SIZE > targetSample.size()) break;

        // 1. Vectorized Conversion & Windowing Loop
        const int32_t* __restrict src = &targetSample[i];
        float* __restrict dest = in.data();
        const float* __restrict win = &hann_11[0]; // Updated to your 256-point table name

        #pragma omp simd
        for (int k = 0; k < FFT_SIZE; ++k) {
            // Convert to float, apply inverse scale, and apply Hann windowing in one step
            dest[k] = (static_cast<float>(src[k]) * inverseScale) * win[k];
        }

        // Execute 256-point FFT
        pocketfft::r2c(shape_in, stride_in, stride_out, axes, pocketfft::FORWARD,
                       in.data(), out.data(), 1.0f);

        FingerprintFrame frame;
        
        // 2. Vectorized Magnitude Loop
        const float* __restrict out_raw = reinterpret_cast<const float*>(out.data());
        float* __restrict frame_raw = reinterpret_cast<float*>(frame.data());
        const float* __restrict weight = &a_weighting[0]; // Uses the exact same coefficients

        // Note: Ensure BINS_TO_KEEP does not exceed 129 to prevent vectorization out-of-bound faults
        #pragma omp simd
        for (int k = 0; k < BINS_TO_KEEP; ++k) {
            float real = out_raw[2 * k];
            float imag = out_raw[2 * k + 1];
            float mag = std::sqrt(real * real + imag * imag);
            frame_raw[k] = mag * weight[k];
        }

        fp[frameIdx] = frame;
    }
    return fp;
}

bool ComputeToneDiffFrameRange(
    const std::vector<std::vector<ToneTuple>>& oldTuples,
    const std::vector<std::vector<ToneTuple>>& newTuples,
    AudioPlayerAL* player,
    size_t totalFrames,
    size_t& outStartFrame,
    size_t& outEndFrame,
    std::vector<ToneTuple>& outAdded,
    std::vector<ToneTuple>& outDeleted)
{
    outAdded.clear();
    outDeleted.clear();

    size_t maxTracks = std::max(oldTuples.size(), newTuples.size());
    for (size_t t = 0; t < maxTracks; ++t) {
        const std::vector<ToneTuple>& oldTrack = (t < oldTuples.size()) ? oldTuples[t] : std::vector<ToneTuple>{};
        const std::vector<ToneTuple>& newTrack = (t < newTuples.size()) ? newTuples[t] : std::vector<ToneTuple>{};

        std::vector<ToneTuple> added, deleted;
        findAddedAndDeleted(oldTrack, newTrack, added, deleted);
        outAdded.insert(outAdded.end(), added.begin(), added.end());
        outDeleted.insert(outDeleted.end(), deleted.begin(), deleted.end());
    }

    size_t totalChanges = outAdded.size() + outDeleted.size();
    if (totalChanges == 0) {
        return false;
    }

    uint64_t minSample = std::numeric_limits<uint64_t>::max();
    uint64_t maxSample = 0;

    auto processTone = [&](const ToneTuple& tone) {
        int64_t startSample = std::get<0>(tone);
        int64_t duration    = std::get<2>(tone);
        int instrument      = std::get<3>(tone);
        int pitchIdx        = std::get<4>(tone);

        uint64_t totalDur = player ? player->GetToneTotalDurationSamples(instrument, pitchIdx, duration)
                                   : static_cast<uint64_t>(duration);

        uint64_t start = (startSample > 0) ? static_cast<uint64_t>(startSample) : 0ULL;
        uint64_t end   = start + totalDur;

        if (start < minSample) minSample = start;
        if (end > maxSample)   maxSample = end;
    };

    for (const auto& tone : outAdded)   processTone(tone);
    for (const auto& tone : outDeleted) processTone(tone);

    if (minSample == std::numeric_limits<uint64_t>::max()) {
        return false;
    }

    // Convert audio sample range to FFT frame index range.
    // FFT_SIZE = 1024, Hop = 512.
    // Frame k covers samples [k*512, k*512 + 1024).
    outStartFrame = (minSample >= 1023) ? (minSample - 1023) / 512 : 0;
    outEndFrame   = (maxSample > 0) ? (maxSample - 1) / 512 + 1 : 0;

    // Use partial fingerprint if the modified frame range affects less than half of the total frames
    if (totalFrames > 0 && (outEndFrame - outStartFrame) >= (totalFrames / 2)) {
        return false;
    }

    return true;
}


double FingerPrintMatch(const std::vector<FingerprintFrame>& f1,
                        const std::vector<FingerprintFrame>& f2,
                        std::vector<double>* outFrameErrors = nullptr,
                        double* outC = nullptr,
                        double fixedC = -1.0)
{
    size_t maxFrames = std::max(f1.size(), f2.size());
    if (outFrameErrors) outFrameErrors->assign(maxFrames, 0.0);
    size_t minFrames = std::min(f1.size(), f2.size());

    // Total elements for the overlapping region
    size_t minElements = minFrames * BINS_TO_KEEP;

    float c = 1.0;
    if (fixedC > 0.0) {
        c = fixedC;
    } else if (minFrames > 0) {
        // Flattened Dot-Product Loop
        float f1sq = 0.0;
        float f1f2 = 0.0;
        
        const float* __restrict flat1 = reinterpret_cast<const float*>(f1[0].data());
        const float* __restrict flat2 = reinterpret_cast<const float*>(f2[0].data());

        
        #pragma omp simd reduction(+:f1sq, f1f2)
        for (size_t i = 0; i < minElements; ++i) {
            float val1 = flat1[i];
            float val2 = flat2[i];
            f1sq += val1 * val1;
            f1f2 += val1 * val2;
        }

        
        if (f1sq > 0.00001) c = f1f2 / f1sq;
    }

    if (outC) *outC = c;

    float sumSquares = 0.0;

    // Main Overlap Loop: Vectorized flat logic for calculation, unflattened for frame errors
    if (minFrames > 0) {
        const float* __restrict flat1 = reinterpret_cast<const float*>(f1[0].data());
        const float* __restrict flat2 = reinterpret_cast<const float*>(f2[0].data());

        if (outFrameErrors) {
            // If the user needs per-frame errors, we keep a nested structure 
            // but the compiler can unroll inner loops easily since BINS_TO_KEEP is constant
            for (size_t i = 0; i < minFrames; ++i) {
                float frameSum = 0.0;
                size_t offset = i * BINS_TO_KEEP;
                
                #pragma omp simd reduction(+:frameSum)
                for (size_t j = 0; j < BINS_TO_KEEP; ++j) {
                    float m1 = std::sqrt((float)flat1[offset + j] * c);
                    float m2 = std::sqrt((float)flat2[offset + j]);
                    float diff = m1 - m2;
                    frameSum += diff * diff;
                }
                sumSquares += frameSum;
                (*outFrameErrors)[i] = frameSum;
            }
        } else {
            // Purely flat single loop when outFrameErrors is null (maximum vectorization)
            
            #pragma omp simd reduction(+:sumSquares)
            for (size_t i = 0; i < minElements; ++i) {
                float m1 = std::sqrt((float)flat1[i] * c);
                float m2 = std::sqrt((float)flat2[i]);
                float diff = m1 - m2;
                sumSquares += diff * diff;
            }
        }
    }

    // Remainder Loop for f1 (if f1 is longer)
    if (f1.size() > minFrames) {
        const float* __restrict flat1 = reinterpret_cast<const float*>(f1[0].data());
        
        if (outFrameErrors) {
            for (size_t i = minFrames; i < f1.size(); ++i) {
                float frameSum = 0.0;
                size_t offset = i * BINS_TO_KEEP;
                #pragma omp simd reduction(+:frameSum)
                for (size_t j = 0; j < BINS_TO_KEEP; ++j) {
                    // Note: sqrt(x * c) * sqrt(x * c) simplifies to just (x * c) if inputs are non-negative
                    float m1 = std::sqrt((float)flat1[offset + j] * c);
                    frameSum += m1 * m1;
                }
                sumSquares += frameSum;
                (*outFrameErrors)[i] = frameSum;
            }
        } else {
            size_t totalElementsF1 = f1.size() * BINS_TO_KEEP;
            #pragma omp simd reduction(+:sumSquares)
            for (size_t i = minElements; i < totalElementsF1; ++i) {
                float m1 = std::sqrt((float)flat1[i] * c);
                sumSquares += m1 * m1;
            }
        }
    }

    // Remainder Loop for f2 (if f2 is longer)
    if (f2.size() > minFrames) {
        const float* __restrict flat2 = reinterpret_cast<const float*>(f2[0].data());
        
        if (outFrameErrors) {
            for (size_t i = minFrames; i < f2.size(); ++i) {
                float frameSum = 0.0;
                size_t offset = i * BINS_TO_KEEP;
                #pragma omp simd reduction(+:frameSum)
                for (size_t j = 0; j < BINS_TO_KEEP; ++j) {
                    float m2 = std::sqrt((float)flat2[offset + j]);
                    frameSum += m2 * m2;
                }
                sumSquares += frameSum;
                (*outFrameErrors)[i] = frameSum;
            }
        } else {
            size_t totalElementsF2 = f2.size() * BINS_TO_KEEP;
            #pragma omp simd reduction(+:sumSquares)
            for (size_t i = minElements; i < totalElementsF2; ++i) {
                float m2 = std::sqrt((float)flat2[i]);
                sumSquares += m2 * m2;
            }
        }
    }

    return std::sqrt(sumSquares);
}

// Generate the ABC text for a ToneList without rendering audio.
// This is cheap compared to the full audio render + FFT pipeline.
std::string ToneListToABC(const ToneList& toneList)
{
    Brute mybrute;
    mybrute.LoadToneList(toneList);
    mybrute.GenerateDefaultConfig(toneList.polyphony);
    std::stringstream bruteConfigStream;
    bruteConfigStream << mybrute.m_MappingText.str();
    mybrute.Transcode(&bruteConfigStream);
    return mybrute.m_ABCText.str();
}

// Generate the list of ToneTuples for a ToneList without rendering audio.
// This is cheap compared to the full audio render + FFT pipeline.
std::vector<std::vector<ToneTuple>> ToneListToToneTuples(const ToneList& toneList)
{
    Brute mybrute;
    return mybrute.Transcode_toneList(toneList, std::vector<size_t>(), std::vector<std::vector<ToneTuple>>{});
}


std::vector<std::vector<ToneTuple>> ToneListToToneTuplesPartial(
    const ToneList& toneList,
    const std::vector<size_t>& dirtyTracks,
    const std::vector<std::vector<ToneTuple>> oldTuple)
{
    // check if all toneList tracks have at least one tone
    size_t mintones = 1;
    for (size_t i = 1; i < toneList.tones.size(); i++)
       if (toneList.tones[i].size() ==0) mintones =0;

    if (( dirtyTracks.size() == 1 ) && (toneList.tones.size()-1 == oldTuple.size()) && (mintones != 0))
    {
  
    ToneList partial;
    partial.notracks = dirtyTracks.size()+1;
    partial.midiinstruments.resize( dirtyTracks.size()+1);
    partial.polyphony.resize( dirtyTracks.size()+1 );
    partial.optimizeEnabled.resize( dirtyTracks.size()+1) ;
    partial.tones.resize(dirtyTracks.size()+1);

    for (size_t i = 0; i < dirtyTracks.size(); i++)
    {
        partial.midiinstruments[ i+1  ] = toneList.midiinstruments[ dirtyTracks[i]  ];
        partial.polyphony[i+1] = toneList.polyphony[dirtyTracks[i]];
        partial.optimizeEnabled[i+1] = toneList.optimizeEnabled[dirtyTracks[i]];
        partial.tones[i+1] = toneList.tones[dirtyTracks[i]];
    }

    Brute mybrute;
    std::vector<std::vector<ToneTuple>> partialTuples = mybrute.Transcode_toneList(partial, std::vector<size_t>(), std::vector<std::vector<ToneTuple>>{});
    

    std::vector<std::vector<ToneTuple>> newTuples;
    newTuples.resize(toneList.tones.size() - 1);  // -1 because track 0 is dummy

    for (size_t t = 0; t < newTuples.size(); ++t) {  // copy over the old tuple
    if (t < oldTuple.size()) {
        newTuples[t] = oldTuple[t];
    }
    }

    // Overwriting the tracks that have been regenerated
    // also check if there is anything wrong
    size_t check = 1;
    for (size_t i = 0; i < dirtyTracks.size(); i++)
    {
        if (partialTuples[i].size() == 0) check = 0;
        newTuples[  dirtyTracks[i]-1  ] = partialTuples[i];
    }

    if ( check == 1 )
    {
       return newTuples;
    }
    else
    {
    Brute mybrute;
    return mybrute.Transcode_toneList(toneList,std::vector<size_t>(), std::vector<std::vector<ToneTuple>>{}) ;
    }

}
else{
    Brute mybrute;
    return mybrute.Transcode_toneList(toneList,std::vector<size_t>(), std::vector<std::vector<ToneTuple>>{}) ;
}
}


std::vector<FingerprintFrame> FingerPrintTuple( std::vector<std::vector<ToneTuple>> tones, AudioPlayerAL * myplayer, const ToneList& toneList)
{
    myplayer->SendToneTuples(tones);

    std::vector<float> generatedAudio = myplayer->GenerateWAVMonof();
    size_t padding = (generatedAudio.size() + 512) % 512;
    for (size_t i = 0; i < padding; i++) generatedAudio.push_back(0.f);


    return FingerPrint(std::move(generatedAudio));
}

// Full fingerprint pipeline: transcode -> render audio -> FFT.
// If abcText is non-empty it is used directly, skipping the Brute transcode step.
std::vector<FingerprintFrame> FingerPrintToneList(const ToneList& toneList,
                                                   AudioPlayerAL* myplayer,
                                                   const std::string& saveAbcAs,
                                                   const std::string& abcText = "")
{
    std::string abc = abcText;
    if (abc.empty()) {
        // Transcode ToneList -> ABC
        Brute mybrute;
        mybrute.LoadToneList(toneList);
        mybrute.GenerateDefaultConfig(toneList.polyphony);
        std::stringstream bruteConfigStream;
        bruteConfigStream << mybrute.m_MappingText.str();
        mybrute.Transcode(&bruteConfigStream);
        abc = mybrute.m_ABCText.str();
    }

    if (!saveAbcAs.empty()) {
        std::ofstream abcFile(saveAbcAs);
        if (abcFile.is_open()) {
            abcFile << abc;
            abcFile.close();
        }
    }

    std::stringstream abcStream;
    abcStream << abc;
    myplayer->SendABC(&abcStream);
    std::vector<float> generatedAudio = myplayer->GenerateWAVMono();

    size_t padding = (generatedAudio.size() + 512) % 512;
    for (size_t i = 0; i < padding; i++) generatedAudio.push_back(0.f);

    return FingerPrint(std::move(generatedAudio));
}






// ---------------------------------------------------------------------------
// Heatmap texture builder
// Returns magnitude in log scale as RGBA pixels (false-colour: blue->green->red)
// ---------------------------------------------------------------------------
// maxFrames: if > 0, crop the fingerprint to at most this many frames (X-axis cap)
static void FingerprintToRGBA(const std::vector<FingerprintFrame>& fp,
                               std::vector<uint8_t>& pixels,
                               int& outW, int& outH,
                               int maxFrames = 0)
{
    const int MAX_W = 2048;
    outH = BINS_TO_KEEP;
    int srcW = (int)fp.size();
    if (maxFrames > 0 && srcW > maxFrames) srcW = maxFrames;
    outW = (int)std::min((size_t)srcW, (size_t)MAX_W);
    pixels.assign((size_t)outW * outH * 4, 0);
    if (outW == 0) return;

    // Precalculate logarithmic bin mapping indices and weights for the Y-scale
    std::vector<int> idxLow(outH);
    std::vector<int> idxHigh(outH);
    std::vector<float> weightLow(outH);
    std::vector<float> weightHigh(outH);

    float denom = (outH > 1) ? (float)(outH - 1) : 1.0f;
    for (int y = 0; y < outH; ++y) {
        float bin_idx = 1.0f * std::pow(100.0f, (float)y / denom);
        int low = static_cast<int>(std::floor(bin_idx));
        int high = low + 1;
        if (high >= BINS_TO_KEEP) {
            low = BINS_TO_KEEP - 1;
            high = BINS_TO_KEEP - 1;
        }
        idxLow[y] = low;
        idxHigh[y] = high;
        float wHigh = bin_idx - low;
        if (wHigh < 0.0f) wHigh = 0.0f;
        if (wHigh > 1.0f) wHigh = 1.0f;
        weightHigh[y] = wHigh;
        weightLow[y] = 1.0f - wHigh;
    }

    // first pass: find max log-magnitude for normalization
    float maxMag = 1e-6f;
    for (int x = 0; x < outW; ++x) {
        for (int y = 0; y < outH; ++y) {
            float val = weightLow[y] * fp[x][idxLow[y]].real + weightHigh[y] * fp[x][idxHigh[y]].real;
            float mag = std::log1p(val);
            if (mag > maxMag) maxMag = mag;
        }
    }

    for (int x = 0; x < outW; ++x) {
        for (int y = 0; y < outH; ++y) {
            float val = weightLow[y] * fp[x][idxLow[y]].real + weightHigh[y] * fp[x][idxHigh[y]].real;
            float mag  = std::log1p(val);
            float t    = mag / maxMag;  // 0..1

            // Inferno-like false colour: black -> purple -> orange -> yellow
            uint8_t r, g, bl;
            if      (t < 0.25f) { float u = t / 0.25f; r = (uint8_t)(40*u);  g = 0;            bl = (uint8_t)(100+155*u); }
            else if (t < 0.5f)  { float u = (t-0.25f)/0.25f; r=(uint8_t)(40+140*u); g=(uint8_t)(20*u); bl=(uint8_t)(255-200*u); }
            else if (t < 0.75f) { float u = (t-0.5f)/0.25f;  r=(uint8_t)(180+75*u); g=(uint8_t)(20+135*u); bl=(uint8_t)(55-55*u); }
            else                { float u = (t-0.75f)/0.25f;  r=255; g=(uint8_t)(155+100*u); bl=0; }

            // Flip Y: row 0 = highest bin (high freq at top), last row = bin 0 (low freq at bottom)
            int row = (outH - 1 - y);
            size_t idx = ((size_t)row * outW + x) * 4;
            pixels[idx+0] = r;
            pixels[idx+1] = g;
            pixels[idx+2] = bl;
            pixels[idx+3] = 255;
        }
    }
}

// Upload / update an OpenGL texture from the fingerprint; returns texture ID.
// maxFrames: crop candidate to at most this many frames (0 = no limit)
static GLuint UploadFingerprintTexture(GLuint texId,
                                        const std::vector<FingerprintFrame>& fp,
                                        int& outW, int& outH,
                                        int maxFrames = 0)
{
    std::vector<uint8_t> pixels;
    FingerprintToRGBA(fp, pixels, outW, outH, maxFrames);
    if (outW == 0 || outH == 0) return texId;

    if (texId == 0)
        glGenTextures(1, &texId);

    glBindTexture(GL_TEXTURE_2D, texId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, outW, outH, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    glBindTexture(GL_TEXTURE_2D, 0);
    return texId;
}

// ---------------------------------------------------------------------------
// Custom widget to plot cost function evolution
// ---------------------------------------------------------------------------
static void DrawCostPlot(const char* label,
                         const std::vector<float>& bestHistory,
                         const std::vector<float>& acceptedHistory,
                         const std::vector<int>& iterationHistory,
                         ImVec2 size)
{
    ImGui::Text("%s", label);
    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    
    // Draw background box
    drawList->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y), ImGui::GetColorU32(ImGuiCol_FrameBg), 4.0f);
    drawList->AddRect(pos, ImVec2(pos.x + size.x, pos.y + size.y), ImGui::GetColorU32(ImGuiCol_Border), 4.0f);
    
    // Reserve the layout space in ImGui
    ImGui::Dummy(size);
    
    if (bestHistory.empty()) {
        ImVec2 textPos = ImVec2(pos.x + 15.f, pos.y + size.y / 2.f - 8.f);
        drawList->AddText(textPos, ImGui::GetColorU32(ImGuiCol_TextDisabled), "No data yet. Start matching to plot.");
        return;
    }
    
    // Find min and max y values (scores/energies)
    float maxVal = -1e30f;
    float minVal = 1e30f;
    for (float v : bestHistory) {
        if (v > maxVal) maxVal = v;
        if (v < minVal) minVal = v;
    }
    for (float v : acceptedHistory) {
        if (v > maxVal) maxVal = v;
        if (v < minVal) minVal = v;
    }
    
    // Add some padding to Y scale
    float diff = maxVal - minVal;
    if (diff < 1e-5f) diff = 1.0f;
    maxVal += diff * 0.05f;
    minVal -= diff * 0.05f;
    if (minVal < 0) minVal = 0;
    
    int nPoints = (int)bestHistory.size();
    
    // Plot borders padding
    float padLeft = 12.0f;
    float padRight = 12.0f;
    float padTop = 15.0f;
    float padBottom = 25.0f;
    
    float graphW = size.x - padLeft - padRight;
    float graphH = size.y - padTop - padBottom;
    
    // Draw grid lines (subtle)
    int gridDivisions = 4;
    ImU32 colGrid = ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 1.0f, 0.04f));
    for (int g = 1; g < gridDivisions; ++g) {
        float t = (float)g / gridDivisions;
        float y = pos.y + padTop + t * graphH;
        drawList->AddLine(ImVec2(pos.x + padLeft, y), ImVec2(pos.x + size.x - padRight, y), colGrid);
        
        float x = pos.x + padLeft + t * graphW;
        drawList->AddLine(ImVec2(x, pos.y + padTop), ImVec2(x, pos.y + size.y - padBottom), colGrid);
    }
    
    auto getGraphPos = [&](int idx, float value) -> ImVec2 {
        float t = (nPoints > 1) ? ((float)idx / (nPoints - 1)) : 0.5f;
        float x = pos.x + padLeft + t * graphW;
        float y_t = (value - minVal) / (maxVal - minVal);
        float y = pos.y + padTop + (1.0f - y_t) * graphH;
        return ImVec2(x, y);
    };
    
    // Draw lines
    ImU32 colBest = ImGui::GetColorU32(ImVec4(0.2f, 0.9f, 0.3f, 0.9f));      // Green
    ImU32 colAccepted = ImGui::GetColorU32(ImVec4(0.9f, 0.9f, 0.1f, 0.7f));  // Yellow
    
    if (nPoints == 1) {
        drawList->AddCircleFilled(getGraphPos(0, bestHistory[0]), 3.0f, colBest);
        drawList->AddCircleFilled(getGraphPos(0, acceptedHistory[0]), 2.0f, colAccepted);
    } else {
        for (int i = 0; i < nPoints - 1; ++i) {
            ImVec2 p1Best = getGraphPos(i, bestHistory[i]);
            ImVec2 p2Best = getGraphPos(i + 1, bestHistory[i + 1]);
            drawList->AddLine(p1Best, p2Best, colBest, 2.0f);
            
            ImVec2 p1Acc = getGraphPos(i, acceptedHistory[i]);
            ImVec2 p2Acc = getGraphPos(i + 1, acceptedHistory[i + 1]);
            drawList->AddLine(p1Acc, p2Acc, colAccepted, 1.0f);
        }
    }
    
    // Draw horizontal grid lines or text indicators (e.g. min, max values)
    char minStr[32], maxStr[32], curStr[64];
    snprintf(minStr, sizeof(minStr), "Min: %.2f", minVal);
    snprintf(maxStr, sizeof(maxStr), "Max: %.2f", maxVal);
    snprintf(curStr, sizeof(curStr), "Best: %.4f", bestHistory.back());
    
    drawList->AddText(ImVec2(pos.x + 12.f, pos.y + 4.f), ImGui::GetColorU32(ImGuiCol_TextDisabled), maxStr);
    drawList->AddText(ImVec2(pos.x + 12.f, pos.y + size.y - padBottom - 14.f), ImGui::GetColorU32(ImGuiCol_TextDisabled), minStr);
    drawList->AddText(ImVec2(pos.x + size.x - 130.f, pos.y + 4.f), ImGui::GetColorU32(ImGuiCol_Text), curStr);
    
    // Draw legend at bottom
    ImVec2 legendPos = ImVec2(pos.x + padLeft, pos.y + size.y - padBottom + 6.0f);
    drawList->AddRectFilled(legendPos, ImVec2(legendPos.x + 8.f, legendPos.y + 8.f), colBest);
    drawList->AddText(ImVec2(legendPos.x + 12.f, legendPos.y - 3.f), ImGui::GetColorU32(ImGuiCol_Text), "Best Model");
    
    legendPos.x += 100.0f;
    drawList->AddRectFilled(legendPos, ImVec2(legendPos.x + 8.f, legendPos.y + 8.f), colAccepted);
    drawList->AddText(ImVec2(legendPos.x + 12.f, legendPos.y - 3.f), ImGui::GetColorU32(ImGuiCol_Text), "Accepted");
    
    // Iterations indicator
    if (!iterationHistory.empty()) {
        char iterStr[32];
        snprintf(iterStr, sizeof(iterStr), "Iter: %d", iterationHistory.back());
        drawList->AddText(ImVec2(pos.x + size.x - 100.f, pos.y + size.y - padBottom + 3.f), ImGui::GetColorU32(ImGuiCol_TextDisabled), iterStr);
    }
}

// ---------------------------------------------------------------------------
// Application state (shared between GUI thread and worker thread)
// ---------------------------------------------------------------------------
struct AppState {
    // --- target ---
    std::string                    wavPath;
    std::string                    tonelistPath;
    std::vector<FingerprintFrame>  targetFingerprint;
    bool                           targetLoaded = false;

    // --- GUI-editable config ---
    // numTracks = number of *real* (non-dummy) tracks; track 0 is always dummy.
    // instruments[t] controls track (t+1) in the ToneList.
    int  numTracks               = 2;   // 2 real tracks → ToneList has 3 entries (0=dummy, 1, 2)
    int  instruments[24]          = {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    bool optimize[24]             = {true, true, true, true, true, true, true, true,
                                     true, true, true, true, true, true, true, true,
                                     true, true, true, true, true, true, true, true};
    int  polyphony[24]            = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int  tempStepIndex            = 0;   // 0 -> 0.2, 1 -> 0.02, 2 -> 0.002, 3 -> 0.0002
    double tempmax                = 0.2; // Initial maximum temperature

    // --- optimizer state (written by worker, read by GUI under mtx) ---
    std::mutex                     mtx;
    ToneList                       bestList;
    double                         bestScore    = 1e28;
    std::vector<FingerprintFrame>  candidateFingerprint;
    int                            iteration    = 0;
    double                         temperature  = 0.2;
    double                         acceptanceRate = 0.0;

    double                         timing_mutation = 0.;
    double                         timing_fingerprinting = 0.;
    double                         timing_fingerprintmatch = 0.;
    double                         timing_total = 0.;
    double                         timing_abcmaking = 0.;

    // --- worker control ---
    std::atomic<bool>              running      { false };
    std::atomic<bool>              stopRequested{ false };
    std::thread                    workerThread;

    // --- OpenGL textures ---
    GLuint texTarget    = 0;
    int    texTargetW   = 0, texTargetH = 0;
    GLuint texCandidate = 0;
    int    texCandW     = 0, texCandH   = 0;
    bool   candidateDirty = false;  // worker sets; GUI clears after upload

    // --- audio ---
    AudioPlayerAL* player        = nullptr; // used exclusively by the optimizer worker
    AudioPlayerAL* previewPlayer = nullptr; // used exclusively for interactive playback
    bool           playbackActive = false;  // true while playing back best candidate

    // --- cost plot history ---
    std::vector<float>             historyBestScore;
    std::vector<float>             historyAcceptedScore;
    std::vector<int>               historyIteration;

    // --- status messages ---
    char   statusMsg[256] = "Drop a WAV file to begin.";

    // --- Sweeping mode config (editable from GUI, read by worker) ---
    bool   use44k              = false; // false = 11.025kHz (_11), true = 44.1kHz
    bool   sweepingMode        = false; // false = full mode, true = sweeping
    int    sweepWindowMult     = 10;    // window size = sweepWindowMult * 2048 samples
    int    sweepGhostMult      = 5;     // ghost area  = sweepGhostMult  * 2048 samples
    int    sweepStepsPerWindow = 10000; // SA steps per window position

    // --- Sweeping mode runtime state (written by worker, read by GUI) ---
    int    sweepCurrentWindow  = 0;
    int    sweepTotalWindows   = 0;
};

// ---------------------------------------------------------------------------
// Worker thread
// ---------------------------------------------------------------------------
void runMatcher(AppState* s)
{
    std::mt19937 gen(815);

    // Build initial ToneList from GUI settings.
    // Track 0 is always the dummy/empty track; real tracks start at index 1.
    ToneList currentList;
    

    // If a tonelist was already pre-loaded (via drag), use it; otherwise build fresh
    {
        std::lock_guard<std::mutex> lk(s->mtx);
        s->historyBestScore.clear();
        s->historyAcceptedScore.clear();
        s->historyIteration.clear();
        if (!s->bestList.tones.empty()) {
            currentList = s->bestList;
        } else {
            int totalTracks = s->numTracks + 1; // +1 for dummy track 0
            currentList.notracks = (uint8_t)totalTracks;
            currentList.tones.resize(totalTracks);
            currentList.midiinstruments.resize(totalTracks, 0);
            currentList.optimizeEnabled.resize(totalTracks, true);
            currentList.polyphony.resize(totalTracks, 1);
            // track 0 left empty (dummy); real tracks 1..numTracks
            for (int t = 0; t < s->numTracks && t < 24; ++t) {
                currentList.midiinstruments[t + 1] = (uint8_t)s->instruments[t];
                currentList.optimizeEnabled[t + 1] = s->optimize[t];
                currentList.polyphony[t + 1]       = (uint8_t)s->polyphony[t];
            }
        }
    }

    // copy target fingerprint locally (read-only from here)
    std::vector<FingerprintFrame> targetFP;
    {
        std::lock_guard<std::mutex> lk(s->mtx);
        targetFP = s->targetFingerprint;
    }

    if (targetFP.empty()) {
        s->running = false;
        return;
    }

    // Initial score
    std::vector<std::vector<ToneTuple>> currentTuple = ToneListToToneTuples(currentList);
    s->player->SendToneTuples(currentTuple);

    std::vector<int32_t> currentAudioI = s->player->GenerateWAVMonoI_11();
    size_t initialPadding = (currentAudioI.size() + 512) % 512;
    for (size_t p = 0; p < initialPadding; p++) currentAudioI.push_back(0);

    std::vector<FingerprintFrame> fp = FingerPrintI_11(currentAudioI); 

    double currentScore = FingerPrintMatch(fp, targetFP);

    // Cache the currently-accepted fingerprint and audio so identical/delta mutations reuse audio & FFT.
    std::vector<FingerprintFrame> currentFP = fp;

    ToneList bestList    = currentList;
    double   bestScore   = currentScore;
    double   temperature = s->tempmax;
    double   accept      = 0.0;

    bool acceptHistory[100] = {false};
    int acceptHistoryIdx = 0;
    int acceptHistoryCount = 0;

    {
        std::lock_guard<std::mutex> lk(s->mtx);
        s->bestList    = bestList;
        s->bestScore   = bestScore;
        s->temperature = temperature;
        s->acceptanceRate = 0.0;
        s->historyBestScore.push_back((float)bestScore);
        s->historyAcceptedScore.push_back((float)currentScore);
        s->historyIteration.push_back(0);
        s->timing_total = 0.;
        s->timing_fingerprinting = 0.;
        s->timing_fingerprintmatch = 0.;
        s->timing_mutation = 0.;
        s->timing_abcmaking = 0.;
    }

    std::vector<double> frameErrors;

    

    double mutation_duration = 0;
    double abcmaking_duration = 0;
    double fingerprintmaking = 0.;
    double fingerprintmatching = 0.;
    double total_duration = 0.;

    int acceptedSinceResync = 0;

    for (int i = 1; !s->stopRequested; ++i) {

        std::chrono::time_point<std::chrono::high_resolution_clock> loop_starting_time = std::chrono::high_resolution_clock::now();

        std::chrono::time_point<std::chrono::high_resolution_clock> chrono_before = std::chrono::high_resolution_clock::now();

        ToneMutation mut = mutate(currentList, gen,
                                  static_cast<uint64_t>(targetFP.size()),10000);

        std::chrono::time_point<std::chrono::high_resolution_clock> chrono_after = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> ST = chrono_after - chrono_before;
        mutation_duration += ST.count();

        // Determine which tracks were dirtied by the mutation and retranscode
        // only those tracks, then patch into a copy of the current tuple vector.
        chrono_before = std::chrono::high_resolution_clock::now();

        std::vector<std::vector<ToneTuple>> canTuple;
        if (mut.action == -1) {
            // No-op mutation: nothing changed, reuse current tuples as-is.
            canTuple = currentTuple;
        } else {
            // Collect the (1 or 2) tracks that the mutation touched.
            std::vector<size_t> dirtyTracks;
            dirtyTracks.push_back(mut.trackId);  // in tuples there is no track 0
            if (mut.action == 3)  // track-switch: both source and destination changed
                dirtyTracks.push_back(mut.targetTrackId);

            // Retranscode only the dirty tracks.
            std::vector<std::vector<ToneTuple>> partialTuple =
                ToneListToToneTuplesPartial(currentList, dirtyTracks, currentTuple);   // check for volume in brute!!! not that partial tracks are shifted up!!!

            // Start from the last accepted tuples and patch in the new data.
            canTuple = partialTuple; 

        }

        chrono_after = std::chrono::high_resolution_clock::now();
        ST = chrono_after - chrono_before;
        abcmaking_duration += ST.count();



        std::vector<FingerprintFrame> candFP;
        std::vector<int32_t> canAudioI;
        double candScore;
        if (canTuple == currentTuple) {
            // Mutation produced no change in the rendered ABC — reuse cache.
            candFP    = currentFP;
            candScore = currentScore;
        } else {
            chrono_before = std::chrono::high_resolution_clock::now();

            size_t startFrame = 0, endFrame = 0;
            std::vector<ToneTuple> addedTones, deletedTones;
            bool usePartialFP = ComputeToneDiffFrameRange(currentTuple, canTuple, s->player, currentFP.size(), startFrame, endFrame, addedTones, deletedTones);

            if (( addedTones.size() < 2 ) &&( deletedTones.size() < 2))
            {
                canAudioI = s->player->ApplyToneDeltaMonoI_11(currentAudioI, addedTones, deletedTones);
            } else
            {
                s->player->SendToneTuples(canTuple);
                canAudioI = s->player->GenerateWAVMonoI_11();
                size_t pad = (canAudioI.size() + 512) % 512;
                 for (size_t p = 0; p < pad; p++) canAudioI.push_back(0);                
            }

            if (usePartialFP) {
                candFP = FingerPrintPartialI_11(canAudioI, currentFP, startFrame, endFrame);

            } else {
                candFP = FingerPrintI_11(canAudioI);
            }

            chrono_after = std::chrono::high_resolution_clock::now();
            ST = chrono_after - chrono_before;
            fingerprintmaking += ST.count();     


            chrono_before = std::chrono::high_resolution_clock::now();

            // candScore = FingerPrintMatch(candFP, targetFP, &frameErrors);
            candScore = FingerPrintMatch(candFP, targetFP);
            chrono_after = std::chrono::high_resolution_clock::now();
            ST = chrono_after - chrono_before;
            fingerprintmatching += ST.count();     
        }

        bool isNewBest = (candScore < bestScore);
        if (isNewBest) {
            // Generate clean full audio & fingerprint for global best to eliminate any roundoff error and verify true score
            s->player->SendToneTuples(canTuple);
            std::vector<int32_t> cleanBestAudioI = s->player->GenerateWAVMonoI_11();
            size_t pad = (cleanBestAudioI.size() + 512) % 512;
            for (size_t p = 0; p < pad; p++) cleanBestAudioI.push_back(0);

            std::vector<FingerprintFrame> testFP = FingerPrintI_11(cleanBestAudioI);
            double cleanScore = FingerPrintMatch(testFP, targetFP);

            if (cleanScore < bestScore) {
                bestScore = cleanScore;
                bestList  = currentList;

                std::lock_guard<std::mutex> lk(s->mtx);
                s->bestList         = bestList;
                s->bestScore        = bestScore;
                s->candidateFingerprint = std::move(testFP);
                s->candidateDirty   = true;
                s->iteration        = i;

                snprintf(s->statusMsg, sizeof(s->statusMsg),
                         "%d  Best: %.4f  T: %.5f IPS: %.5f%%", i, bestScore, temperature, (100*s->timing_total/i) );
            }
        }

        bool accepted = false;
        if (candScore <= currentScore ||
            (double)rand() / RAND_MAX < std::exp((currentScore - candScore) / temperature))
        {
            accepted = true;
            accept += 1.0;
            acceptedSinceResync++;
            currentScore = candScore;
            currentFP    = std::move(candFP);
            currentTuple = canTuple;
            if (!canAudioI.empty()) {
                currentAudioI = std::move(canAudioI);
            }

            // Periodic re-sync of SA walk state to prevent accumulation of floating point roundoff errors
             if (acceptedSinceResync >= 1000) {
               s->player->SendToneTuples(currentTuple);
               currentAudioI = s->player->GenerateWAVMonoI_11();
               size_t pad = (currentAudioI.size() + 512) % 512;
               for (size_t p = 0; p < pad; p++) currentAudioI.push_back(0);

               currentFP = FingerPrintI_11(currentAudioI);
               currentScore = FingerPrintMatch(currentFP, targetFP);
               acceptedSinceResync = 0;
            }
        } else {
            revertMutation(currentList, mut);
        }

        // Update sliding window of last 100 steps
        if (acceptHistory[acceptHistoryIdx]) {
            acceptHistoryCount--;
        }
        acceptHistory[acceptHistoryIdx] = accepted;
        if (accepted) {
            acceptHistoryCount++;
        }
        acceptHistoryIdx = (acceptHistoryIdx + 1) % 100;

        double currentAcceptanceRate = (double)acceptHistoryCount / std::min(i, 100) * 100.0;

        if (i % 500 == 0) {
            std::lock_guard<std::mutex> lk(s->mtx);
            s->historyBestScore.push_back((float)bestScore);
            s->historyAcceptedScore.push_back((float)currentScore);
            s->historyIteration.push_back(i);
        }

        ST = std::chrono::high_resolution_clock::now() - loop_starting_time;
        total_duration += ST.count();

        if (i % 100 == 0) {
            if (accept > 3)  temperature *= coolingrate;
            if ( i% reheat_intervall == 0) temperature = s->tempmax;
           
           // if (accept < .01)  temperature *= 1.5;
            accept = 0;
            std::lock_guard<std::mutex> lk(s->mtx);
            s->temperature = temperature;
            s->iteration   = i;
            s->acceptanceRate = currentAcceptanceRate;

            // every 100 steps accumulated times are added
            s->timing_mutation  += mutation_duration;
            s->timing_fingerprinting += fingerprintmaking;
            s->timing_fingerprintmatch += fingerprintmatching;
            s->timing_abcmaking += abcmaking_duration;
            s->timing_total += total_duration;

            mutation_duration = 0.;
            fingerprintmaking = 0.;
            fingerprintmatching = 0.;
            abcmaking_duration = 0.;
            total_duration = 0.;
        }

        // Periodic checkpoint save
        if (i % 1000000 == 0) {
            ToneList saveList;
            {
                std::lock_guard<std::mutex> lk(s->mtx);
                saveList = s->bestList;
            }
            SaveToneListToFile(saveList, "checkpoint.tonelist");
            FingerPrintToneList(saveList, s->player, "checkpoint.abc");
            std::cout << "Checkpoint saved at iteration " << i << "\n";
        }
    }

    s->running = false;
}

// ---------------------------------------------------------------------------
// Worker thread (44.1 kHz engine)
// ---------------------------------------------------------------------------
void runMatcher_44(AppState* s)
{
    std::mt19937 gen(815);

    // Build initial ToneList from GUI settings.
    // Track 0 is always the dummy/empty track; real tracks start at index 1.
    ToneList currentList;
    

    // If a tonelist was already pre-loaded (via drag), use it; otherwise build fresh
    {
        std::lock_guard<std::mutex> lk(s->mtx);
        s->historyBestScore.clear();
        s->historyAcceptedScore.clear();
        s->historyIteration.clear();
        if (!s->bestList.tones.empty()) {
            currentList = s->bestList;
        } else {
            int totalTracks = s->numTracks + 1; // +1 for dummy track 0
            currentList.notracks = (uint8_t)totalTracks;
            currentList.tones.resize(totalTracks);
            currentList.midiinstruments.resize(totalTracks, 0);
            currentList.optimizeEnabled.resize(totalTracks, true);
            currentList.polyphony.resize(totalTracks, 1);
            // track 0 left empty (dummy); real tracks 1..numTracks
            for (int t = 0; t < s->numTracks && t < 24; ++t) {
                currentList.midiinstruments[t + 1] = (uint8_t)s->instruments[t];
                currentList.optimizeEnabled[t + 1] = s->optimize[t];
                currentList.polyphony[t + 1]       = (uint8_t)s->polyphony[t];
            }
        }
    }

    // copy target fingerprint locally (read-only from here)
    std::vector<FingerprintFrame> targetFP;
    {
        std::lock_guard<std::mutex> lk(s->mtx);
        targetFP = s->targetFingerprint;
    }

    if (targetFP.empty()) {
        s->running = false;
        return;
    }

    // Initial score
    std::vector<std::vector<ToneTuple>> currentTuple = ToneListToToneTuples(currentList);
    s->player->SendToneTuples(currentTuple);

    std::vector<int32_t> currentAudioI = s->player->GenerateWAVMonoI();
    size_t initialPadding = (currentAudioI.size() + 512) % 512;
    for (size_t p = 0; p < initialPadding; p++) currentAudioI.push_back(0);

    std::vector<FingerprintFrame> fp = FingerPrintI(currentAudioI); 

    double currentScore = FingerPrintMatch(fp, targetFP);

    // Cache the currently-accepted fingerprint and audio so identical/delta mutations reuse audio & FFT.
    std::vector<FingerprintFrame> currentFP = fp;

    ToneList bestList    = currentList;
    double   bestScore   = currentScore;
    double   temperature = s->tempmax;
    double   accept      = 0.0;

    bool acceptHistory[100] = {false};
    int acceptHistoryIdx = 0;
    int acceptHistoryCount = 0;

    {
        std::lock_guard<std::mutex> lk(s->mtx);
        s->bestList    = bestList;
        s->bestScore   = bestScore;
        s->temperature = temperature;
        s->acceptanceRate = 0.0;
        s->historyBestScore.push_back((float)bestScore);
        s->historyAcceptedScore.push_back((float)currentScore);
        s->historyIteration.push_back(0);
        s->timing_total = 0.;
        s->timing_fingerprinting = 0.;
        s->timing_fingerprintmatch = 0.;
        s->timing_mutation = 0.;
        s->timing_abcmaking = 0.;
    }

    std::vector<double> frameErrors;

    

    double mutation_duration = 0;
    double abcmaking_duration = 0;
    double fingerprintmaking = 0.;
    double fingerprintmatching = 0.;
    double total_duration = 0.;

    // Stats: clear previous run records
    if (g_statsEnabled) {
        g_statsRecords.clear();
        g_statsRecords.reserve(1 << 20); // pre-allocate ~1M entries
    }

    int acceptedSinceResync = 0;

    for (int i = 1; !s->stopRequested; ++i) {

        std::chrono::time_point<std::chrono::high_resolution_clock> loop_starting_time = std::chrono::high_resolution_clock::now();

        std::chrono::time_point<std::chrono::high_resolution_clock> chrono_before = std::chrono::high_resolution_clock::now();

        ToneMutation mut = mutate(currentList, gen,
                                  static_cast<uint64_t>(targetFP.size()),10000);

        std::chrono::time_point<std::chrono::high_resolution_clock> chrono_after = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> ST = chrono_after - chrono_before;
        mutation_duration += ST.count();

        // Determine which tracks were dirtied by the mutation and retranscode
        // only those tracks, then patch into a copy of the current tuple vector.
        chrono_before = std::chrono::high_resolution_clock::now();

        std::vector<std::vector<ToneTuple>> canTuple;
        if (mut.action == -1) {
            // No-op mutation: nothing changed, reuse current tuples as-is.
            canTuple = currentTuple;
        } else {
            // Collect the (1 or 2) tracks that the mutation touched.
            std::vector<size_t> dirtyTracks;
            dirtyTracks.push_back(mut.trackId);  // in tuples there is no track 0
            if (mut.action == 3)  // track-switch: both source and destination changed
                dirtyTracks.push_back(mut.targetTrackId);

            // Retranscode only the dirty tracks.
            std::vector<std::vector<ToneTuple>> partialTuple =
                ToneListToToneTuplesPartial(currentList, dirtyTracks, currentTuple);   // check for volume in brute!!! not that partial tracks are shifted up!!!

            // Start from the last accepted tuples and patch in the new data.
            canTuple = partialTuple; 

        }

        chrono_after = std::chrono::high_resolution_clock::now();
        ST = chrono_after - chrono_before;
        abcmaking_duration += ST.count();



        std::vector<FingerprintFrame> candFP;
        std::vector<int32_t> canAudioI;
        double candScore;
        if (canTuple == currentTuple) {
            // Mutation produced no change in the rendered ABC — reuse cache.
            candFP    = currentFP;
            candScore = currentScore;
        } else {
            chrono_before = std::chrono::high_resolution_clock::now();

            size_t startFrame = 0, endFrame = 0;
            std::vector<ToneTuple> addedTones, deletedTones;
            bool usePartialFP = ComputeToneDiffFrameRange(currentTuple, canTuple, s->player, currentFP.size(), startFrame, endFrame, addedTones, deletedTones);

            if (( addedTones.size() < 2 ) &&( deletedTones.size() < 2))
            {
                canAudioI = s->player->ApplyToneDeltaMonoI(currentAudioI, addedTones, deletedTones);
            } else
            {
                s->player->SendToneTuples(canTuple);
                canAudioI = s->player->GenerateWAVMonoI();
                size_t pad = (canAudioI.size() + 512) % 512;
                 for (size_t p = 0; p < pad; p++) canAudioI.push_back(0);                
            }

            if (usePartialFP) {
                candFP = FingerPrintPartialI(canAudioI, currentFP, startFrame, endFrame);

            } else {
                candFP = FingerPrintI(canAudioI);
            }

            chrono_after = std::chrono::high_resolution_clock::now();
            ST = chrono_after - chrono_before;
            fingerprintmaking += ST.count();     


            chrono_before = std::chrono::high_resolution_clock::now();

            // candScore = FingerPrintMatch(candFP, targetFP, &frameErrors);
            candScore = FingerPrintMatch(candFP, targetFP);
            chrono_after = std::chrono::high_resolution_clock::now();
            ST = chrono_after - chrono_before;
            fingerprintmatching += ST.count();     
        }

        bool isNewBest = (candScore < bestScore);
        bool confirmedNewBest = false;
        if (isNewBest) {
            // Generate clean full audio & fingerprint for global best to eliminate any roundoff error and verify true score
            s->player->SendToneTuples(canTuple);
            std::vector<int32_t> cleanBestAudioI = s->player->GenerateWAVMonoI();
            size_t pad = (cleanBestAudioI.size() + 512) % 512;
            for (size_t p = 0; p < pad; p++) cleanBestAudioI.push_back(0);

            std::vector<FingerprintFrame> testFP = FingerPrintI(cleanBestAudioI);
            double cleanScore = FingerPrintMatch(testFP, targetFP);

            if (cleanScore < bestScore) {
                confirmedNewBest = true;
                bestScore = cleanScore;
                bestList  = currentList;

                std::lock_guard<std::mutex> lk(s->mtx);
                s->bestList         = bestList;
                s->bestScore        = bestScore;
                s->candidateFingerprint = std::move(testFP);
                s->candidateDirty   = true;
                s->iteration        = i;

                snprintf(s->statusMsg, sizeof(s->statusMsg),
                         "%d  Best: %.4f  T: %.5f IPS: %.5f%%", i, bestScore, temperature, (100*s->timing_total/i) );
            }
        }

        // Capture energy difference BEFORE acceptance updates currentScore
        const double energyDiff = candScore - currentScore;

        bool accepted = false;
        if (candScore <= currentScore ||
            (double)rand() / RAND_MAX < std::exp((currentScore - candScore) / temperature))
        {
            accepted = true;
            accept += 1.0;
            acceptedSinceResync++;
            currentScore = candScore;
            currentFP    = std::move(candFP);
            currentTuple = canTuple;
            if (!canAudioI.empty()) {
                currentAudioI = std::move(canAudioI);
            }

            // Periodic re-sync of SA walk state to prevent accumulation of floating point roundoff errors
             if (acceptedSinceResync >= 1000) {
               s->player->SendToneTuples(currentTuple);
               currentAudioI = s->player->GenerateWAVMonoI();
               size_t pad = (currentAudioI.size() + 512) % 512;
               for (size_t p = 0; p < pad; p++) currentAudioI.push_back(0);

               currentFP = FingerPrintI(currentAudioI);
               currentScore = FingerPrintMatch(currentFP, targetFP);
               acceptedSinceResync = 0;
            }
        } else {
            revertMutation(currentList, mut);
        }

        // Collect statistics (only for real mutations, not no-ops)
        if (g_statsEnabled && mut.action != -1) {
            StatRecord rec;
            rec.moveType   = mut.action;
            rec.energyDiff = energyDiff; // candScore - prevCurrentScore (before acceptance)
            rec.accepted   = accepted;
            rec.newBest    = confirmedNewBest;
            g_statsRecords.push_back(rec);
        }

        // Update sliding window of last 100 steps
        if (acceptHistory[acceptHistoryIdx]) {
            acceptHistoryCount--;
        }
        acceptHistory[acceptHistoryIdx] = accepted;
        if (accepted) {
            acceptHistoryCount++;
        }
        acceptHistoryIdx = (acceptHistoryIdx + 1) % 100;

        double currentAcceptanceRate = (double)acceptHistoryCount / std::min(i, 100) * 100.0;

        if (i % 500 == 0) {
            std::lock_guard<std::mutex> lk(s->mtx);
            s->historyBestScore.push_back((float)bestScore);
            s->historyAcceptedScore.push_back((float)currentScore);
            s->historyIteration.push_back(i);
        }

        ST = std::chrono::high_resolution_clock::now() - loop_starting_time;
        total_duration += ST.count();

        if (i % 100 == 0) {
            if (accept > 3)  temperature *= coolingrate;
            if ( i% reheat_intervall == 0) temperature = s->tempmax;
           
           // if (accept < .01)  temperature *= 1.5;
            accept = 0;
            std::lock_guard<std::mutex> lk(s->mtx);
            s->temperature = temperature;
            s->iteration   = i;
            s->acceptanceRate = currentAcceptanceRate;

            // every 100 steps accumulated times are added
            s->timing_mutation  += mutation_duration;
            s->timing_fingerprinting += fingerprintmaking;
            s->timing_fingerprintmatch += fingerprintmatching;
            s->timing_abcmaking += abcmaking_duration;
            s->timing_total += total_duration;

            mutation_duration = 0.;
            fingerprintmaking = 0.;
            fingerprintmatching = 0.;
            abcmaking_duration = 0.;
            total_duration = 0.;
        }

        // Periodic checkpoint save
        if (i % 1000000 == 0) {
            ToneList saveList;
            {
                std::lock_guard<std::mutex> lk(s->mtx);
                saveList = s->bestList;
            }
            SaveToneListToFile(saveList, "checkpoint.tonelist");
            FingerPrintToneList(saveList, s->player, "checkpoint.abc");
            std::cout << "Checkpoint saved at iteration " << i << "\n";
        }
    }

    s->running = false;
}

// ---------------------------------------------------------------------------
// Sweeping mode worker thread
// ---------------------------------------------------------------------------
void runMatcherSweep(AppState* s)
{
    std::mt19937 gen(815);

    // Read sweeping config under lock, then release
    uint64_t windowSizeSamples, ghostSizeSamples;
    int stepsPerWindow;
    {
        std::lock_guard<std::mutex> lk(s->mtx);
        windowSizeSamples = (uint64_t)s->sweepWindowMult * 2048;
        ghostSizeSamples  = (uint64_t)s->sweepGhostMult  * 2048;
        stepsPerWindow    = s->sweepStepsPerWindow;
    }


    for (int mumu = 1; !s->stopRequested; ++mumu) {
    // Each sweep step moves the window right by half its size (multiple of 1024)
    uint64_t stepSize = windowSizeSamples / 2;

    // Build / load the full ToneList (same logic as runMatcher)
    ToneList fullList;
    {
        std::lock_guard<std::mutex> lk(s->mtx);
        s->historyBestScore.clear();
        s->historyAcceptedScore.clear();
        s->historyIteration.clear();
        if (!s->bestList.tones.empty()) {
            fullList = s->bestList;
        } else {
            int totalTracks = s->numTracks + 1;
            fullList.notracks = (uint8_t)totalTracks;
            fullList.tones.resize(totalTracks);
            fullList.midiinstruments.resize(totalTracks, 0);
            fullList.optimizeEnabled.resize(totalTracks, true);
            fullList.polyphony.resize(totalTracks, 1);
            for (int t = 0; t < s->numTracks && t < 24; ++t) {
                fullList.midiinstruments[t + 1] = (uint8_t)s->instruments[t];
                fullList.optimizeEnabled[t + 1] = s->optimize[t];
                fullList.polyphony[t + 1]       = (uint8_t)s->polyphony[t];
            }
        }
    }

    // Copy target fingerprint locally (read-only from here)
    std::vector<FingerprintFrame> targetFP;
    {
        std::lock_guard<std::mutex> lk(s->mtx);
        targetFP = s->targetFingerprint;
    }
    if (targetFP.empty()) { s->running = false; return; }

    // Total sample span implied by the target FP (hop = 512 samples)
    uint64_t totalSamples = (uint64_t)targetFP.size() * 512;

    // Build the list of window start positions
    std::vector<uint64_t> windowStarts;
    for (uint64_t ws = 0; ws < totalSamples; ws += stepSize)
        windowStarts.push_back(ws);
    int totalWindows = (int)windowStarts.size();

    {
        std::lock_guard<std::mutex> lk(s->mtx);
        s->sweepTotalWindows  = totalWindows;
        s->sweepCurrentWindow = 0;
        s->temperature        = s->tempmax * std::pow( coolingrate  ,mumu*stepsPerWindow);
        snprintf(s->statusMsg, sizeof(s->statusMsg),
                 "Sweep starting: %d windows, %.2f s each",
                 totalWindows, (float)windowSizeSamples / 44100.0f);
    }

    int globalIteration = 0;

    // Check if fullList contains any tones before the sweep starts
    size_t totalToneCount = 0;
    for (const auto& track : fullList.tones) {
        totalToneCount += track.size();
    }

    double predeterminedC = -1.0;
    if (totalToneCount > 0) {
        std::vector<std::vector<ToneTuple>> fullTuples = ToneListToToneTuples(fullList);
        s->player->SendToneTuples(fullTuples);
        std::vector<int32_t> fullAudioI = s->player->GenerateWAVMonoI_11();
        {
            size_t pad = (fullAudioI.size() + 512) % 512;
            for (size_t p = 0; p < pad; p++) fullAudioI.push_back(0);
        }
        std::vector<FingerprintFrame> fullFP = FingerPrintI_11(fullAudioI);
        FingerPrintMatch(fullFP, targetFP, nullptr, &predeterminedC);
    }

    for (int wIdx = 0; wIdx < totalWindows && !s->stopRequested; ++wIdx) {
        uint64_t windowStart = windowStarts[wIdx];
        uint64_t windowEnd   = windowStart + windowSizeSamples;
        if (windowEnd > totalSamples) windowEnd = totalSamples;
        // Align windowEnd to 1024-sample FFT boundary
        windowEnd = (windowEnd / 1024) * 1024;
        if (windowEnd <= windowStart) windowEnd = windowStart + 1024;

        // --- Extract window ToneList (active zone + ghost context) ---
        uint64_t windowOffset = 0;
        ToneList windowList = extractWindowToneList(
            fullList, windowStart, windowEnd, ghostSizeSamples, windowOffset);

        // Active window bounds in local (shifted) coordinates
        uint64_t localWindowStart = windowStart - windowOffset;
        uint64_t localWindowEnd   = windowEnd   - windowOffset;

        // --- Extract matching target FP range: central window + post-ghost ---
        // Pre-ghost is rendered for audio context (tones from before can bleed in)
        // but is NOT scored — only the central window and post-ghost are matched.
        // This means mutations in the central window are evaluated including any
        // audio that bleeds forward past windowEnd, which is the desired behaviour.
        // Mutations are still restricted to [localWindowStart, localWindowEnd) by windowedMutate.
        size_t fpCenterStart        = std::min(windowStart / 512, targetFP.size());
        size_t fpMatchEnd           = std::min((windowEnd + ghostSizeSamples + 511) / 512, targetFP.size());
        size_t numMatchFrames       = (fpMatchEnd > fpCenterStart) ? (fpMatchEnd - fpCenterStart) : 0;
        size_t candCenterStartFrame = (windowStart - windowOffset) / 512;

        // Lambda to score candidate FP against target FP for central window + post-ghost
        auto computeWindowScore = [&](const std::vector<FingerprintFrame>& fullCandFP) -> double {
            if (numMatchFrames == 0 || candCenterStartFrame >= fullCandFP.size()) return 1e28;
            size_t actualMatchFrames = std::min(numMatchFrames, fullCandFP.size() - candCenterStartFrame);

            std::vector<FingerprintFrame> matchedCandFP(
                fullCandFP.begin() + (ptrdiff_t)candCenterStartFrame,
                fullCandFP.begin() + (ptrdiff_t)(candCenterStartFrame + actualMatchFrames));

            std::vector<FingerprintFrame> matchedTargetFP(
                targetFP.begin() + (ptrdiff_t)fpCenterStart,
                targetFP.begin() + (ptrdiff_t)(fpCenterStart + actualMatchFrames));

            return FingerPrintMatch(matchedCandFP, matchedTargetFP, nullptr, nullptr, predeterminedC);
        };

        // --- Initial render of the full capture zone (ghost-area | center window | ghost-area) ---
        std::vector<std::vector<ToneTuple>> currentTuple = ToneListToToneTuples(windowList);
        s->player->SendToneTuples(currentTuple);
        std::vector<int32_t> currentAudioI = s->player->GenerateWAVMonoI_11();
        {
            size_t pad = (currentAudioI.size() + 512) % 512;
            for (size_t p = 0; p < pad; p++) currentAudioI.push_back(0);
        }
        std::vector<FingerprintFrame> currentFP    = FingerPrintI_11(currentAudioI);
        double                        currentScore  = computeWindowScore(currentFP);

        ToneList bestWindowList  = windowList;
        double   bestWindowScore = currentScore;
        double   temperature     = s->tempmax; // reset temperature for each window
        double   accept          = 0.0;
        int      acceptedSinceResync = 0;

        bool acceptHistWin[100] = {false};
        int  acceptHistWinIdx   = 0;
        int  acceptHistWinCount = 0;

        // --- Mini SA loop for this window position ---
        for (int i = 1; i <= stepsPerWindow && !s->stopRequested; ++i) {

            ToneMutation mut = windowedMutate(
                windowList, gen, localWindowStart, localWindowEnd, 10000);

            // Retranscode only the dirty tracks
            std::vector<std::vector<ToneTuple>> canTuple;
            if (mut.action == -1) {
                canTuple = currentTuple;
            } else {
                std::vector<size_t> dirtyTracks;
                dirtyTracks.push_back(mut.trackId);
                if (mut.action == 3)
                    dirtyTracks.push_back(mut.targetTrackId);
                canTuple = ToneListToToneTuplesPartial(windowList, dirtyTracks, currentTuple);
            }

            std::vector<FingerprintFrame> candFP;
            std::vector<int32_t>          canAudioI;
            double                        candScore;

            if (canTuple == currentTuple) {
                candFP    = currentFP;
                candScore = currentScore;
            } else {
                size_t startFrame = 0, endFrame = 0;
                std::vector<ToneTuple> addedTones, deletedTones;
                bool usePartialFP = ComputeToneDiffFrameRange(
                    currentTuple, canTuple, s->player,
                    currentFP.size(), startFrame, endFrame, addedTones, deletedTones);

                if ((addedTones.size() < 2) && (deletedTones.size() < 2)) {
                    canAudioI = s->player->ApplyToneDeltaMonoI_11(
                        currentAudioI, addedTones, deletedTones);
                } else {
                    s->player->SendToneTuples(canTuple);
                    canAudioI = s->player->GenerateWAVMonoI_11();
                    size_t pad = (canAudioI.size() + 512) % 512;
                    for (size_t p = 0; p < pad; p++) canAudioI.push_back(0);
                }

                if (usePartialFP) {
                    candFP = FingerPrintPartialI_11(canAudioI, currentFP, startFrame, endFrame);
                } else {
                    candFP = FingerPrintI_11(canAudioI);
                }
                candScore = computeWindowScore(candFP);
            }

            // Update local window best (with clean-render verification)
            if (candScore < bestWindowScore) {
                s->player->SendToneTuples(canTuple);
                std::vector<int32_t> cleanAudioI = s->player->GenerateWAVMonoI_11();
                {
                    size_t pad = (cleanAudioI.size() + 512) % 512;
                    for (size_t p = 0; p < pad; p++) cleanAudioI.push_back(0);
                }
                std::vector<FingerprintFrame> testFP     = FingerPrintI_11(cleanAudioI);
                double                        cleanScore  = computeWindowScore(testFP);
                if (cleanScore < bestWindowScore) {
                    bestWindowScore = cleanScore;
                    bestWindowList  = windowList;
                }
            }

            // Metropolis acceptance criterion
            bool accepted = false;
            if (candScore <= currentScore ||
                (double)rand() / RAND_MAX < std::exp((currentScore - candScore) / temperature))
            {
                accepted = true;
                accept  += 1.0;
                acceptedSinceResync++;
                currentScore = candScore;
                currentFP    = std::move(candFP);
                currentTuple = canTuple;
                if (!canAudioI.empty())
                    currentAudioI = std::move(canAudioI);

                // Periodic resync to prevent floating-point drift
                if (acceptedSinceResync >= 1000) {
                    s->player->SendToneTuples(currentTuple);
                    currentAudioI = s->player->GenerateWAVMonoI_11();
                    {
                        size_t pad = (currentAudioI.size() + 512) % 512;
                        for (size_t p = 0; p < pad; p++) currentAudioI.push_back(0);
                    }
                    currentFP    = FingerPrintI_11(currentAudioI);
                    currentScore = computeWindowScore(currentFP);
                    acceptedSinceResync = 0;
                }
            } else {
                revertMutation(windowList, mut);
            }

            // Sliding-window acceptance rate
            if (acceptHistWin[acceptHistWinIdx]) acceptHistWinCount--;
            acceptHistWin[acceptHistWinIdx] = accepted;
            if (accepted) acceptHistWinCount++;
            acceptHistWinIdx = (acceptHistWinIdx + 1) % 100;

            // Temperature schedule (identical to full mode)
            if (i % 100 == 0) {
                if (accept > 3) temperature *= coolingrate;
                accept = 0;
            }

            ++globalIteration;
        } // end mini SA loop

        // Merge the best local result back into fullList
        mergeWindowToneList(fullList, bestWindowList, windowStart, windowEnd, windowOffset);

        // Report progress to GUI
        {
            std::lock_guard<std::mutex> lk(s->mtx);
            s->sweepCurrentWindow = wIdx + 1;
            s->iteration          = globalIteration;
            s->temperature        = temperature;
            s->historyBestScore.push_back((float)bestWindowScore);
            s->historyAcceptedScore.push_back((float)bestWindowScore);
            s->historyIteration.push_back(globalIteration);
            snprintf(s->statusMsg, sizeof(s->statusMsg),
                     "Sweeping: window %d/%d  Window score: %.4f",
                     wIdx + 1, totalWindows, bestWindowScore);
        }
    } // end window sweep loop

    // Final full-length evaluation after all windows are processed
    if (!s->stopRequested) {
        std::vector<std::vector<ToneTuple>> finalTuples = ToneListToToneTuples(fullList);
        s->player->SendToneTuples(finalTuples);
        std::vector<int32_t> finalAudioI = s->player->GenerateWAVMonoI_11();
        {
            size_t pad = (finalAudioI.size() + 512) % 512;
            for (size_t p = 0; p < pad; p++) finalAudioI.push_back(0);
        }
        std::vector<FingerprintFrame> finalFP    = FingerPrintI_11(finalAudioI);
        double                        finalScore  = FingerPrintMatch(finalFP, targetFP);

        std::lock_guard<std::mutex> lk(s->mtx);
        s->bestList               = fullList;
        s->bestScore              = finalScore;
        s->candidateFingerprint   = finalFP;
        s->candidateDirty         = true;
        snprintf(s->statusMsg, sizeof(s->statusMsg),
                 "Sweep complete.  Full score: %.4f", finalScore);
    }
    s->running = false;
}
}


// ---------------------------------------------------------------------------
// Sweeping mode worker thread (44.1 kHz engine)
// ---------------------------------------------------------------------------
void runMatcherSweep_44(AppState* s)
{
    std::mt19937 gen(815);

    // Read sweeping config under lock, then release
    uint64_t windowSizeSamples, ghostSizeSamples;
    int stepsPerWindow;
    {
        std::lock_guard<std::mutex> lk(s->mtx);
        windowSizeSamples = (uint64_t)s->sweepWindowMult * 2048;
        ghostSizeSamples  = (uint64_t)s->sweepGhostMult  * 2048;
        stepsPerWindow    = s->sweepStepsPerWindow;
    }


    for (int mumu = 1; !s->stopRequested; ++mumu) {
    // Each sweep step moves the window right by half its size (multiple of 1024)
    uint64_t stepSize = windowSizeSamples / 2;

    // Build / load the full ToneList (same logic as runMatcher)
    ToneList fullList;
    {
        std::lock_guard<std::mutex> lk(s->mtx);
        s->historyBestScore.clear();
        s->historyAcceptedScore.clear();
        s->historyIteration.clear();
        if (!s->bestList.tones.empty()) {
            fullList = s->bestList;
        } else {
            int totalTracks = s->numTracks + 1;
            fullList.notracks = (uint8_t)totalTracks;
            fullList.tones.resize(totalTracks);
            fullList.midiinstruments.resize(totalTracks, 0);
            fullList.optimizeEnabled.resize(totalTracks, true);
            fullList.polyphony.resize(totalTracks, 1);
            for (int t = 0; t < s->numTracks && t < 24; ++t) {
                fullList.midiinstruments[t + 1] = (uint8_t)s->instruments[t];
                fullList.optimizeEnabled[t + 1] = s->optimize[t];
                fullList.polyphony[t + 1]       = (uint8_t)s->polyphony[t];
            }
        }
    }

    // Copy target fingerprint locally (read-only from here)
    std::vector<FingerprintFrame> targetFP;
    {
        std::lock_guard<std::mutex> lk(s->mtx);
        targetFP = s->targetFingerprint;
    }
    if (targetFP.empty()) { s->running = false; return; }

    // Total sample span implied by the target FP (hop = 512 samples)
    uint64_t totalSamples = (uint64_t)targetFP.size() * 512;

    // Build the list of window start positions
    std::vector<uint64_t> windowStarts;
    for (uint64_t ws = 0; ws < totalSamples; ws += stepSize)
        windowStarts.push_back(ws);
    int totalWindows = (int)windowStarts.size();

    {
        std::lock_guard<std::mutex> lk(s->mtx);
        s->sweepTotalWindows  = totalWindows;
        s->sweepCurrentWindow = 0;
        s->temperature        = s->tempmax * std::pow( coolingrate  ,mumu*stepsPerWindow);
        snprintf(s->statusMsg, sizeof(s->statusMsg),
                 "Sweep starting: %d windows, %.2f s each",
                 totalWindows, (float)windowSizeSamples / 44100.0f);
    }

    int globalIteration = 0;

    // Check if fullList contains any tones before the sweep starts
    size_t totalToneCount = 0;
    for (const auto& track : fullList.tones) {
        totalToneCount += track.size();
    }

    double predeterminedC = -1.0;
    if (totalToneCount > 0) {
        std::vector<std::vector<ToneTuple>> fullTuples = ToneListToToneTuples(fullList);
        s->player->SendToneTuples(fullTuples);
        std::vector<int32_t> fullAudioI = s->player->GenerateWAVMonoI();
        {
            size_t pad = (fullAudioI.size() + 512) % 512;
            for (size_t p = 0; p < pad; p++) fullAudioI.push_back(0);
        }
        std::vector<FingerprintFrame> fullFP = FingerPrintI(fullAudioI);
        FingerPrintMatch(fullFP, targetFP, nullptr, &predeterminedC);
    }

    for (int wIdx = 0; wIdx < totalWindows && !s->stopRequested; ++wIdx) {
        uint64_t windowStart = windowStarts[wIdx];
        uint64_t windowEnd   = windowStart + windowSizeSamples;
        if (windowEnd > totalSamples) windowEnd = totalSamples;
        // Align windowEnd to 1024-sample FFT boundary
        windowEnd = (windowEnd / 1024) * 1024;
        if (windowEnd <= windowStart) windowEnd = windowStart + 1024;

        // --- Extract window ToneList (active zone + ghost context) ---
        uint64_t windowOffset = 0;
        ToneList windowList = extractWindowToneList(
            fullList, windowStart, windowEnd, ghostSizeSamples, windowOffset);

        // Active window bounds in local (shifted) coordinates
        uint64_t localWindowStart = windowStart - windowOffset;
        uint64_t localWindowEnd   = windowEnd   - windowOffset;

        // --- Extract matching target FP range: central window + post-ghost ---
        // Pre-ghost is rendered for audio context (tones from before can bleed in)
        // but is NOT scored — only the central window and post-ghost are matched.
        // This means mutations in the central window are evaluated including any
        // audio that bleeds forward past windowEnd, which is the desired behaviour.
        // Mutations are still restricted to [localWindowStart, localWindowEnd) by windowedMutate.
        size_t fpCenterStart        = std::min(windowStart / 512, targetFP.size());
        size_t fpMatchEnd           = std::min((windowEnd + ghostSizeSamples + 511) / 512, targetFP.size());
        size_t numMatchFrames       = (fpMatchEnd > fpCenterStart) ? (fpMatchEnd - fpCenterStart) : 0;
        size_t candCenterStartFrame = (windowStart - windowOffset) / 512;

        // Lambda to score candidate FP against target FP for central window + post-ghost
        auto computeWindowScore = [&](const std::vector<FingerprintFrame>& fullCandFP) -> double {
            if (numMatchFrames == 0 || candCenterStartFrame >= fullCandFP.size()) return 1e28;
            size_t actualMatchFrames = std::min(numMatchFrames, fullCandFP.size() - candCenterStartFrame);

            std::vector<FingerprintFrame> matchedCandFP(
                fullCandFP.begin() + (ptrdiff_t)candCenterStartFrame,
                fullCandFP.begin() + (ptrdiff_t)(candCenterStartFrame + actualMatchFrames));

            std::vector<FingerprintFrame> matchedTargetFP(
                targetFP.begin() + (ptrdiff_t)fpCenterStart,
                targetFP.begin() + (ptrdiff_t)(fpCenterStart + actualMatchFrames));

            return FingerPrintMatch(matchedCandFP, matchedTargetFP, nullptr, nullptr, predeterminedC);
        };

        // --- Initial render of the full capture zone (ghost-area | center window | ghost-area) ---
        std::vector<std::vector<ToneTuple>> currentTuple = ToneListToToneTuples(windowList);
        s->player->SendToneTuples(currentTuple);
        std::vector<int32_t> currentAudioI = s->player->GenerateWAVMonoI();
        {
            size_t pad = (currentAudioI.size() + 512) % 512;
            for (size_t p = 0; p < pad; p++) currentAudioI.push_back(0);
        }
        std::vector<FingerprintFrame> currentFP    = FingerPrintI(currentAudioI);
        double                        currentScore  = computeWindowScore(currentFP);

        ToneList bestWindowList  = windowList;
        double   bestWindowScore = currentScore;
        double   temperature     = s->tempmax; // reset temperature for each window
        double   accept          = 0.0;
        int      acceptedSinceResync = 0;

        bool acceptHistWin[100] = {false};
        int  acceptHistWinIdx   = 0;
        int  acceptHistWinCount = 0;

        // --- Mini SA loop for this window position ---
        for (int i = 1; i <= stepsPerWindow && !s->stopRequested; ++i) {

            ToneMutation mut = windowedMutate(
                windowList, gen, localWindowStart, localWindowEnd, 10000);

            // Retranscode only the dirty tracks
            std::vector<std::vector<ToneTuple>> canTuple;
            if (mut.action == -1) {
                canTuple = currentTuple;
            } else {
                std::vector<size_t> dirtyTracks;
                dirtyTracks.push_back(mut.trackId);
                if (mut.action == 3)
                    dirtyTracks.push_back(mut.targetTrackId);
                canTuple = ToneListToToneTuplesPartial(windowList, dirtyTracks, currentTuple);
            }

            std::vector<FingerprintFrame> candFP;
            std::vector<int32_t>          canAudioI;
            double                        candScore;

            if (canTuple == currentTuple) {
                candFP    = currentFP;
                candScore = currentScore;
            } else {
                size_t startFrame = 0, endFrame = 0;
                std::vector<ToneTuple> addedTones, deletedTones;
                bool usePartialFP = ComputeToneDiffFrameRange(
                    currentTuple, canTuple, s->player,
                    currentFP.size(), startFrame, endFrame, addedTones, deletedTones);

                if ((addedTones.size() < 2) && (deletedTones.size() < 2)) {
                    canAudioI = s->player->ApplyToneDeltaMonoI(
                        currentAudioI, addedTones, deletedTones);
                } else {
                    s->player->SendToneTuples(canTuple);
                    canAudioI = s->player->GenerateWAVMonoI();
                    size_t pad = (canAudioI.size() + 512) % 512;
                    for (size_t p = 0; p < pad; p++) canAudioI.push_back(0);
                }

                if (usePartialFP) {
                    candFP = FingerPrintPartialI(canAudioI, currentFP, startFrame, endFrame);
                } else {
                    candFP = FingerPrintI(canAudioI);
                }
                candScore = computeWindowScore(candFP);
            }

            // Update local window best (with clean-render verification)
            if (candScore < bestWindowScore) {
                s->player->SendToneTuples(canTuple);
                std::vector<int32_t> cleanAudioI = s->player->GenerateWAVMonoI();
                {
                    size_t pad = (cleanAudioI.size() + 512) % 512;
                    for (size_t p = 0; p < pad; p++) cleanAudioI.push_back(0);
                }
                std::vector<FingerprintFrame> testFP     = FingerPrintI(cleanAudioI);
                double                        cleanScore  = computeWindowScore(testFP);
                if (cleanScore < bestWindowScore) {
                    bestWindowScore = cleanScore;
                    bestWindowList  = windowList;
                }
            }

            // Metropolis acceptance criterion
            bool accepted = false;
            if (candScore <= currentScore ||
                (double)rand() / RAND_MAX < std::exp((currentScore - candScore) / temperature))
            {
                accepted = true;
                accept  += 1.0;
                acceptedSinceResync++;
                currentScore = candScore;
                currentFP    = std::move(candFP);
                currentTuple = canTuple;
                if (!canAudioI.empty())
                    currentAudioI = std::move(canAudioI);

                // Periodic resync to prevent floating-point drift
                if (acceptedSinceResync >= 1000) {
                    s->player->SendToneTuples(currentTuple);
                    currentAudioI = s->player->GenerateWAVMonoI();
                    {
                        size_t pad = (currentAudioI.size() + 512) % 512;
                        for (size_t p = 0; p < pad; p++) currentAudioI.push_back(0);
                    }
                    currentFP    = FingerPrintI(currentAudioI);
                    currentScore = computeWindowScore(currentFP);
                    acceptedSinceResync = 0;
                }
            } else {
                revertMutation(windowList, mut);
            }

            // Sliding-window acceptance rate
            if (acceptHistWin[acceptHistWinIdx]) acceptHistWinCount--;
            acceptHistWin[acceptHistWinIdx] = accepted;
            if (accepted) acceptHistWinCount++;
            acceptHistWinIdx = (acceptHistWinIdx + 1) % 100;

            // Temperature schedule (identical to full mode)
            if (i % 100 == 0) {
                if (accept > 3) temperature *= coolingrate;
                accept = 0;
            }

            ++globalIteration;
        } // end mini SA loop

        // Merge the best local result back into fullList
        mergeWindowToneList(fullList, bestWindowList, windowStart, windowEnd, windowOffset);

        // Report progress to GUI
        {
            std::lock_guard<std::mutex> lk(s->mtx);
            s->sweepCurrentWindow = wIdx + 1;
            s->iteration          = globalIteration;
            s->temperature        = temperature;
            s->historyBestScore.push_back((float)bestWindowScore);
            s->historyAcceptedScore.push_back((float)bestWindowScore);
            s->historyIteration.push_back(globalIteration);
            snprintf(s->statusMsg, sizeof(s->statusMsg),
                     "Sweeping: window %d/%d  Window score: %.4f",
                     wIdx + 1, totalWindows, bestWindowScore);
        }
    } // end window sweep loop

    // Final full-length evaluation after all windows are processed
    if (!s->stopRequested) {
        std::vector<std::vector<ToneTuple>> finalTuples = ToneListToToneTuples(fullList);
        s->player->SendToneTuples(finalTuples);
        std::vector<int32_t> finalAudioI = s->player->GenerateWAVMonoI();
        {
            size_t pad = (finalAudioI.size() + 512) % 512;
            for (size_t p = 0; p < pad; p++) finalAudioI.push_back(0);
        }
        std::vector<FingerprintFrame> finalFP    = FingerPrintI(finalAudioI);
        double                        finalScore  = FingerPrintMatch(finalFP, targetFP);

        std::lock_guard<std::mutex> lk(s->mtx);
        s->bestList               = fullList;
        s->bestScore              = finalScore;
        s->candidateFingerprint   = finalFP;
        s->candidateDirty         = true;
        snprintf(s->statusMsg, sizeof(s->statusMsg),
                 "Sweep complete.  Full score: %.4f", finalScore);
    }
    }
    s->running = false;
}

// ---------------------------------------------------------------------------
// GLFW drop callback — stored globally so the lambda can capture AppState*
// ---------------------------------------------------------------------------
static AppState* g_appState = nullptr;

static void dropCallback(GLFWwindow* /*window*/, int count, const char** paths)
{
    if (!g_appState || count < 1) return;
    AppState& s = *g_appState;

    for (int n = 0; n < count; ++n) {
        std::string p(paths[n]);
        // Determine extension
        auto ext = p.substr(p.find_last_of('.') + 1);
        // lowercase
        for (char& c : ext) c = (char)std::tolower((unsigned char)c);

        if (ext == "wav") {
            if (s.running) {
                snprintf(s.statusMsg, sizeof(s.statusMsg), "Stop matching first before loading a new WAV.");
                continue;
            }
            s.wavPath = p;
            std::vector<float> raw = LoadTargetSample(p);
            if (raw.empty()) {
                snprintf(s.statusMsg, sizeof(s.statusMsg), "Failed to load WAV: %s", p.c_str());
                continue;
            }
            auto fp = FingerPrint(std::move(raw));
            {
                std::lock_guard<std::mutex> lk(s.mtx);
                s.targetFingerprint = std::move(fp);
                s.targetLoaded = true;
            }
            // Upload texture (must be done on main thread — we're in the GLFW callback which IS main thread)
            s.texTarget = UploadFingerprintTexture(s.texTarget, s.targetFingerprint,
                                                    s.texTargetW, s.texTargetH);
            snprintf(s.statusMsg, sizeof(s.statusMsg),
                     "Loaded WAV: %s  (%d frames)", p.c_str(), (int)s.targetFingerprint.size());

        } else if (ext == "tonelist") {
            if (s.running) {
                snprintf(s.statusMsg, sizeof(s.statusMsg), "Stop matching first before loading a ToneList.");
                continue;
            }
            ToneList tl = LoadToneListFromFile(p);
            if (tl.tones.empty()) {
                snprintf(s.statusMsg, sizeof(s.statusMsg), "Failed to load ToneList: %s", p.c_str());
                continue;
            }
            s.tonelistPath = p;
            {
                std::lock_guard<std::mutex> lk(s.mtx);
                s.bestList = tl;
                // Sync GUI controls: track 0 is dummy, so real tracks = tones.size()-1
                int realTracks = (int)tl.tones.size() - 1;
                if (realTracks < 1) realTracks = 1;
                if (realTracks > 24) realTracks = 24;
                s.numTracks = realTracks;
                // instruments[i] controls ToneList track i+1
                for (int t = 0; t < realTracks && t < 24; ++t) {
                    s.instruments[t] = ((t + 1) < (int)tl.midiinstruments.size())
                                       ? tl.midiinstruments[t + 1] : 0;
                    s.optimize[t] = ((t + 1) < (int)tl.optimizeEnabled.size())
                                    ? (bool)tl.optimizeEnabled[t + 1] : true;
                    s.polyphony[t] = ((t + 1) < (int)tl.polyphony.size())
                                     ? (int)tl.polyphony[t + 1] : 1;
                }
            }
            snprintf(s.statusMsg, sizeof(s.statusMsg),
                     "Loaded ToneList: %s  (%d tracks)", p.c_str(), (int)tl.tones.size());
        } else {
            snprintf(s.statusMsg, sizeof(s.statusMsg), "Unknown file type (need .wav or .tonelist): %s", p.c_str());
        }
    }
}

// ---------------------------------------------------------------------------
// main — GLFW + ImGui event loop
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // --- Parse command-line flags ---
    for (int a = 1; a < argc; ++a) {
        if (std::string(argv[a]) == "--stats") {
            g_statsEnabled = true;
            std::cout << "[stats] Statistics collection enabled. Data will be saved to stats.tsv on Stop.\n";
        }
    }

    hann = createHannWindow(1024);
    hann_11 = createHannWindow(256);
    a_weighting = createAWeighting(101);

    // --- Audio engines ---
    // myEngine is used solely by the optimizer worker thread for WAV rendering.
    // previewEngine is used solely for interactive playback from the GUI.
    // Keeping them separate eliminates any data race on myabc.
    AudioPlayerAL myEngine;
    myEngine.Initialize(100.0f, 64);

    AudioPlayerAL previewEngine;
    previewEngine.Initialize(100.0f, 64);

    // --- App state ---
    AppState appState;
    appState.player        = &myEngine;
    appState.previewPlayer = &previewEngine;
    g_appState = &appState;

    // --- GLFW ---
    if (!glfwInit()) {
        std::cerr << "GLFW init failed\n";
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1400, 820, "SampleMatcher", nullptr, nullptr);
    if (!window) {
        std::cerr << "GLFW window creation failed\n";
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);  // vsync

    glfwSetDropCallback(window, dropCallback);

    // --- ImGui ---
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Style
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding    = 6.0f;
    style.FrameRounding     = 4.0f;
    style.GrabRounding      = 4.0f;
    style.ItemSpacing       = ImVec2(8, 6);
    style.FramePadding      = ImVec2(6, 4);
    // Accent colour: teal
    style.Colors[ImGuiCol_Button]        = ImVec4(0.10f, 0.55f, 0.55f, 1.0f);
    style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.15f, 0.70f, 0.70f, 1.0f);
    style.Colors[ImGuiCol_ButtonActive]  = ImVec4(0.05f, 0.40f, 0.40f, 1.0f);
    style.Colors[ImGuiCol_Header]        = ImVec4(0.10f, 0.45f, 0.45f, 1.0f);
    style.Colors[ImGuiCol_CheckMark]     = ImVec4(0.20f, 0.90f, 0.90f, 1.0f);
    style.Colors[ImGuiCol_SliderGrab]    = ImVec4(0.20f, 0.75f, 0.75f, 1.0f);
    style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.08f, 0.35f, 0.40f, 1.0f);

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    const char* instrNames[] = {
        "Horn", "Clarinet", "Flute", "Bagpipes",
        "Bardic Fiddle",   "Pibgorn", "Basic Bassoon","Theorbo"
    };

    // --- Main loop ---
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // Upload updated candidate texture on main thread if worker flagged it.
        // Cap candidate X to the same number of frames as the target texture.
        {
            std::lock_guard<std::mutex> lk(appState.mtx);
            if (appState.candidateDirty && !appState.candidateFingerprint.empty()) {
                appState.texCandidate = UploadFingerprintTexture(
                    appState.texCandidate,
                    appState.candidateFingerprint,
                    appState.texCandW, appState.texCandH,
                    appState.texTargetW);  // cap to target frame count
                appState.candidateDirty = false;
            }
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Full-screen window
        int dispW, dispH;
        glfwGetFramebufferSize(window, &dispW, &dispH);
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2((float)dispW, (float)dispH));
        ImGui::Begin("SampleMatcher", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove     | ImGuiWindowFlags_NoBringToFrontOnFocus);

        // ---- Title bar ----
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 0.9f, 0.9f, 1.f));
        ImGui::SetWindowFontScale(1.4f);
        ImGui::Text("  SampleMatcher");
        ImGui::SetWindowFontScale(1.0f);
        ImGui::PopStyleColor();
        ImGui::Separator();

        // ---- Drop zone hint ----
        ImGui::Spacing();
        ImGui::TextDisabled("Drop a .wav file to load target sample.  "
                            "Drop a .tonelist file to resume a previous match.");
        ImGui::Spacing();

        // ---- Status ----
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.8f, 0.85f, 0.5f, 1.f));
        ImGui::TextWrapped("%s", appState.statusMsg);
        ImGui::PopStyleColor();
        ImGui::Separator();
        ImGui::Spacing();

        // ---- Config panel ----
        ImGui::BeginGroup();
        ImGui::Text("Configuration");
        ImGui::Spacing();

        bool matcherRunning = appState.running.load();

        ImGui::BeginDisabled(matcherRunning);

        ImGui::SetNextItemWidth(140.f);
        ImGui::SliderInt("ABC Tracks", &appState.numTracks, 1, 24);

        static const double tempStepValues[4] = { 0.2, 0.02, 0.002, 0.0002 };
        static const char* tempStepLabels[4]  = { "0.2", "0.02", "0.002", "0.0002" };
        ImGui::SetNextItemWidth(140.f);
        if (ImGui::SliderInt("Temperature", &appState.tempStepIndex, 0, 3, tempStepLabels[appState.tempStepIndex])) {
            if (appState.tempStepIndex < 0) appState.tempStepIndex = 0;
            if (appState.tempStepIndex > 3) appState.tempStepIndex = 3;
            appState.tempmax = tempStepValues[appState.tempStepIndex];
        }

        ImGui::Spacing();
        ImGui::Text("Instruments per track (track 1 .. %d):", appState.numTracks);
        ImGui::Indent(12.f);
        for (int t = 0; t < appState.numTracks && t < 24; ++t) {
            char chk_label[32];
            snprintf(chk_label, sizeof(chk_label), "##opt%d", t);
            ImGui::Checkbox(chk_label, &appState.optimize[t]);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Optimize Track %d", t + 1);
            ImGui::SameLine();

            char label[32];
            // Display as "Track 1", "Track 2", … (skip dummy track 0)
            snprintf(label, sizeof(label), "Track %d##instr%d", t + 1, t);
            ImGui::SetNextItemWidth(200.f);
            ImGui::SliderInt(label, &appState.instruments[t], 0, 7);
            if (appState.instruments[t] < 8) {
                ImGui::SameLine();
                ImGui::TextDisabled("(%s)", instrNames[appState.instruments[t]]);
            }
            ImGui::SameLine(380.f);
            char poly_label[32];
            snprintf(poly_label, sizeof(poly_label), "Polyphony##poly%d", t);
            ImGui::SetNextItemWidth(100.f);
            ImGui::SliderInt(poly_label, &appState.polyphony[t], 1, 6);
        }
        ImGui::Unindent(12.f);

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::Text("Audio Engine:");
        ImGui::SameLine();
        if (ImGui::RadioButton("11.025 kHz##engine", !appState.use44k)) appState.use44k = false;
        ImGui::SameLine();
        if (ImGui::RadioButton("44.1 kHz##engine", appState.use44k)) appState.use44k = true;

        ImGui::Spacing();
        ImGui::Text("Optimization Mode:");
        ImGui::SameLine();
        if (ImGui::RadioButton("Full##mode", !appState.sweepingMode)) appState.sweepingMode = false;
        ImGui::SameLine();
        if (ImGui::RadioButton("Sweeping##mode", appState.sweepingMode)) appState.sweepingMode = true;

        if (appState.sweepingMode) {
            ImGui::Indent(12.f);
            ImGui::SetNextItemWidth(160.f);
            ImGui::SliderInt("Window (x2048 smp)##swwin", &appState.sweepWindowMult, 2, 200);
            ImGui::SameLine();
            ImGui::TextDisabled("(%.2f s)", (float)(appState.sweepWindowMult * 2048) / 44100.0f);
            ImGui::SetNextItemWidth(160.f);
            ImGui::SliderInt("Ghost  (x2048 smp)##swghost", &appState.sweepGhostMult, 0, 100);
            ImGui::SameLine();
            ImGui::TextDisabled("(%.2f s)", (float)(appState.sweepGhostMult * 2048) / 44100.0f);
            ImGui::SetNextItemWidth(160.f);
            ImGui::SliderInt("Steps/Window##swsteps", &appState.sweepStepsPerWindow, 500, 10000);
            ImGui::Unindent(12.f);
        }

        ImGui::EndDisabled();
        ImGui::EndGroup();

        ImGui::SameLine(0.f, 40.f);

        ImGui::BeginGroup();
        std::vector<float> localBest;
        std::vector<float> localAccepted;
        std::vector<int> localIteration;
        {
            std::lock_guard<std::mutex> lk(appState.mtx);
            localBest = appState.historyBestScore;
            localAccepted = appState.historyAcceptedScore;
            localIteration = appState.historyIteration;
        }
        DrawCostPlot("Cost Function Evolution", localBest, localAccepted, localIteration, ImVec2(500.f, 180.f));
        ImGui::EndGroup();

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // ---- Buttons ----
        // Match
        if (!matcherRunning) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.10f, 0.60f, 0.20f, 1.f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.15f, 0.80f, 0.30f, 1.f));
            if (ImGui::Button("  Match  ", ImVec2(110, 34))) {
                if (!appState.targetLoaded) {
                    snprintf(appState.statusMsg, sizeof(appState.statusMsg),
                             "Load a WAV file first (drag & drop).");
                } else {
                    // Sync ToneList structure and metadata with GUI settings (tracks, instruments, optimization)
                    {
                        std::lock_guard<std::mutex> lk(appState.mtx);
                        bool wasEmpty = appState.bestList.tones.empty();
                        int total = appState.numTracks + 1;
                        appState.bestList.notracks = (uint8_t)total;
                        appState.bestList.tones.resize(total);
                        appState.bestList.midiinstruments.resize(total, 0);
                        appState.bestList.optimizeEnabled.resize(total, true);
                        appState.bestList.polyphony.resize(total, 1);
                        for (int t = 0; t < appState.numTracks && t < 24; ++t) {
                            appState.bestList.midiinstruments[t + 1] = (uint8_t)appState.instruments[t];
                            appState.bestList.optimizeEnabled[t + 1] = appState.optimize[t];
                            appState.bestList.polyphony[t + 1]       = (uint8_t)appState.polyphony[t];
                        }
                        if (wasEmpty) {
                            appState.bestScore = 1e18;
                        }
                    }
                    appState.stopRequested = false;
                    appState.running       = true;
                    if (appState.workerThread.joinable())
                        appState.workerThread.join();
                    appState.workerThread = std::thread(
                        appState.use44k
                            ? (appState.sweepingMode ? runMatcherSweep_44 : runMatcher_44)
                            : (appState.sweepingMode ? runMatcherSweep : runMatcher),
                        &appState);
                    snprintf(appState.statusMsg, sizeof(appState.statusMsg), "Matching started…");
                }
            }
            ImGui::PopStyleColor(2);
        } else {
            // Show a pulsing indicator while running
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.55f, 0.20f, 0.05f, 1.f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.70f, 0.25f, 0.05f, 1.f));
            if (ImGui::Button("  Stop   ", ImVec2(110, 34))) {
                appState.stopRequested = true;
                snprintf(appState.statusMsg, sizeof(appState.statusMsg), "Stopping…");
                // Save stats once the worker thread finishes (join happens on next Match click or at exit).
                // We schedule the save by setting a flag; the actual join+save happens below.
                if (g_statsEnabled) {
                    // Worker will naturally stop; save stats after join.
                    // We join inline here (brief stall) so the file is written immediately.
                    if (appState.workerThread.joinable()) {
                        appState.workerThread.join();
                        appState.running = false;
                    }
                    saveStats("stats.tsv");
                    snprintf(appState.statusMsg, sizeof(appState.statusMsg),
                             "Stopped. Stats saved to stats.tsv (%zu records).",
                             g_statsRecords.size());
                }
            }
            ImGui::PopStyleColor(2);
        }

        ImGui::SameLine(0.f, 16.f);

        // ---- Play / Stop playback of best candidate ----
        {
            bool hasCandidate;
            {
                std::lock_guard<std::mutex> lk(appState.mtx);
                hasCandidate = !appState.bestList.tones.empty();
            }
            ImGui::BeginDisabled(!hasCandidate);
            if (!appState.playbackActive) {
                ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.15f, 0.40f, 0.70f, 1.f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.20f, 0.55f, 0.90f, 1.f));
                if (ImGui::Button("  Play   ", ImVec2(100, 34))) {
                    ToneList playList;
                    {
                        std::lock_guard<std::mutex> lk(appState.mtx);
                        playList = appState.bestList;
                    }

                    // Render ABC from best list and start playback
                    //smf::MidiFile mf = createMidiFromToneList(playList, 960);
                    //std::stringstream ms;
                    //mf.write(ms);
                    //ms.seekg(0);


                    Brute b;
                    //b.LoadMidi(ms);
                    b.LoadToneList(playList);
                    b.GenerateDefaultConfig(appState.bestList.polyphony);
                    std::stringstream cs;
                    cs << b.m_MappingText.str();
                    b.Transcode(&cs);
                    std::stringstream as;
                    as << b.m_ABCText.str();
                    appState.previewPlayer->Stop();
                    appState.previewPlayer->SendABC(&as);
                    appState.previewPlayer->Play();
                    appState.playbackActive = true;
                    snprintf(appState.statusMsg, sizeof(appState.statusMsg), "Playing best candidate…");
                }
                ImGui::PopStyleColor(2);
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.45f, 0.15f, 0.55f, 1.f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.60f, 0.20f, 0.70f, 1.f));
                if (ImGui::Button(" Stop Ply", ImVec2(100, 34))) {
                    appState.previewPlayer->Stop();
                    appState.playbackActive = false;
                    snprintf(appState.statusMsg, sizeof(appState.statusMsg), "Playback stopped.");
                }
                ImGui::PopStyleColor(2);
                // Auto-reset when player finishes
                if (appState.previewPlayer->Finished()) {
                    appState.previewPlayer->Stop();
                    appState.playbackActive = false;
                }
            }
            ImGui::EndDisabled();
        }

        ImGui::SameLine(0.f, 16.f);

        // Export ToneList
        ImGui::BeginDisabled(matcherRunning);
        if (ImGui::Button("Export ToneList", ImVec2(150, 34))) {
            ToneList save;
            {
                std::lock_guard<std::mutex> lk(appState.mtx);
                save = appState.bestList;
            }
            if (!save.tones.empty()) {
                SaveToneListToFile(save, "export.tonelist");
                snprintf(appState.statusMsg, sizeof(appState.statusMsg), "Saved: export.tonelist");
            } else {
                snprintf(appState.statusMsg, sizeof(appState.statusMsg), "Nothing to export yet.");
            }
        }
        ImGui::SameLine(0.f, 8.f);
        // Export ABC
        if (ImGui::Button("Export ABC", ImVec2(120, 34))) {
            ToneList save;
            {
                std::lock_guard<std::mutex> lk(appState.mtx);
                save = appState.bestList;
            }
            if (!save.tones.empty()) {
                FingerPrintToneList(save, appState.player, "export.abc");
                snprintf(appState.statusMsg, sizeof(appState.statusMsg), "Saved: export.abc");
            } else {
                snprintf(appState.statusMsg, sizeof(appState.statusMsg), "Nothing to export yet.");
            }
        }
        ImGui::EndDisabled();

        // Stats line
        ImGui::SameLine(0.f, 24.f);
        {
            double  bs;
            int     it;
            double  tmp;
            double  accRate;
            double  total;
            double  abcmaking;
            double  fingerprinting;
            double  fingerprintmatch;
            {
                std::lock_guard<std::mutex> lk(appState.mtx);
                bs  = appState.bestScore;
                it  = appState.iteration;
                tmp = appState.temperature;
                accRate = appState.acceptanceRate;
                total = appState.timing_total;
                abcmaking = appState.timing_abcmaking;
                fingerprinting = appState.timing_fingerprinting;
                fingerprintmatch = appState.timing_fingerprintmatch;
            }
            if (it > 0)
                ImGui::Text("It: %d   Best: %.4f   T: %.5f   10K It/s: %.1f ABC: %.1f FP: %.1f FPM: %.1f", it, bs, tmp, 
                    10000.0*total/it, 10000.0*abcmaking/it, 10000.0*fingerprinting/it, 10000.0*fingerprintmatch/it);
        }
        {
            int curW, totW;
            {
                std::lock_guard<std::mutex> lk(appState.mtx);
                curW = appState.sweepCurrentWindow;
                totW = appState.sweepTotalWindows;
            }
            if (appState.sweepingMode && totW > 0) {
                ImGui::SameLine(0.f, 24.f);
                ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.5f, 1.f),
                    "Window: %d / %d", curW, totW);
            }
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // ---- Fingerprint heatmaps ----
        float availW = ImGui::GetContentRegionAvail().x;
        float availH = ImGui::GetContentRegionAvail().y - 30.f;
        float halfW  = (availW - 16.f) * 0.5f;

        // --- Target fingerprint ---
        ImGui::BeginChild("##target_panel", ImVec2(halfW, availH), true);
        ImGui::TextColored(ImVec4(0.3f, 0.9f, 0.9f, 1.f), "Target fingerprint");
        if (appState.texTarget != 0) {
            float imgH = std::min(availH - 30.f, (float)appState.texTargetH * 4.5f);
            ImGui::Image((ImTextureID)(intptr_t)appState.texTarget,
                         ImVec2(halfW - 16.f, imgH),
                         ImVec2(0, 0), ImVec2(1, 1));
        } else {
            ImGui::TextDisabled("(drop a WAV file)");
        }
        ImGui::EndChild();

        ImGui::SameLine(0.f, 8.f);

        // --- Candidate fingerprint ---
        ImGui::BeginChild("##cand_panel", ImVec2(halfW, availH), true);
        ImGui::TextColored(ImVec4(0.9f, 0.75f, 0.2f, 1.f), "Best match fingerprint");
        if (appState.texCandidate != 0) {
            float imgH = std::min(availH - 30.f, (float)appState.texCandH * 4.5f);
            ImGui::Image((ImTextureID)(intptr_t)appState.texCandidate,
                         ImVec2(halfW - 16.f, imgH),
                         ImVec2(0, 0), ImVec2(1, 1));
        } else {
            ImGui::TextDisabled("(start matching to see result)");
        }
        ImGui::EndChild();

        ImGui::End();  // SampleMatcher window

        // --- Render ---
        ImGui::Render();
        glViewport(0, 0, dispW, dispH);
        glClearColor(0.10f, 0.10f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // --- Shutdown ---
    appState.stopRequested = true;
    if (appState.workerThread.joinable())
        appState.workerThread.join();

    if (appState.texTarget    != 0) glDeleteTextures(1, &appState.texTarget);
    if (appState.texCandidate != 0) glDeleteTextures(1, &appState.texCandidate);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
