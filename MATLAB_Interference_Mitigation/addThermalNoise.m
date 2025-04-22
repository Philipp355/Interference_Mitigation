% addThermalNoise.m
function noisy_signal = addThermalNoise(signal, noise_power)
    noise_variance = 10^(noise_power/10);
    noise = sqrt(noise_variance/2) * (randn(size(signal)) + 1j*randn(size(signal)));
    noisy_signal = signal + noise;
end
