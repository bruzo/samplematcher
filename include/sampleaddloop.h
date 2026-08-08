       /*
              #pragma omp simd
                for (uint64_t s = 0; s < looplimit; ++s)
                {
                    dest[s] += static_cast<int32_t>(static_cast<float>(src[s]) * toneGain);
                }


*/
                                
                

int16_t q15HalfGain = static_cast<int16_t>((toneGain * 0.5f) * 32767.0f);

#pragma omp simd
for (uint64_t s = 0; s < looplimit; ++s)
{
    // vpmulhrsw handles this entire line in hardware
    int16_t halfScaled = static_cast<int16_t>((static_cast<int32_t>(src[s]) * q15HalfGain + 16384) >> 15);
    
    // Shift left by 1 to multiply by 2, then accumulate into 32-bit dest
    dest[s] += (static_cast<int32_t>(halfScaled) << 1);
}
                
                
                