function T = snr_to_ber_ter_sim()
    rng(1);

    snrDbVec = 0:1:5;

    M = 4;
    k = log2(M);

    tokenBits = 5;

    maxBitsPerSNR = 5e7;
    minErrsPerSNR = 2e4;

    frameBits = 20000;

    useGPU = true;
    useParfor = true;

    if useParfor
        p = gcp('nocreate');
        if ~isempty(p) && isa(p,'parallel.ThreadPool')
            delete(p);
            p = [];
        end
        if isempty(p)
            parpool('local');
        end
    end

    berVec = zeros(size(snrDbVec));
    terVec = zeros(size(snrDbVec));
    bitCountVec = zeros(size(snrDbVec));
    errCountVec = zeros(size(snrDbVec));

    if useParfor
        parfor i = 1:numel(snrDbVec)
            [berVec(i), bitCountVec(i), errCountVec(i)] = runOneSNR( ...
                snrDbVec(i), M, k, frameBits, maxBitsPerSNR, minErrsPerSNR, useGPU);
            terVec(i) = 1 - (1 - berVec(i))^tokenBits;
        end
    else
        for i = 1:numel(snrDbVec)
            [berVec(i), bitCountVec(i), errCountVec(i)] = runOneSNR( ...
                snrDbVec(i), M, k, frameBits, maxBitsPerSNR, minErrsPerSNR, useGPU);
            terVec(i) = 1 - (1 - berVec(i))^tokenBits;
        end
    end

    T = table(snrDbVec(:), berVec(:), terVec(:), bitCountVec(:), errCountVec(:), ...
        'VariableNames', {'snr_db','ber','ter','n_bits','n_errs'});

    writetable(T, 'snr_ber_ter.csv');
    disp(T);
end

function [ber, nBits, nErrs] = runOneSNR(snrDb, M, k, frameBits, maxBits, minErrs, useGPU)
    %     IEEE 802.11a standard
    trellis = poly2trellis(7, [133 171]);
    tbl = 34;

    enc = comm.ConvolutionalEncoder(trellis, 'TerminationMethod','Truncated');
    dec = comm.ViterbiDecoder(trellis, ...
        'InputFormat','Unquantized', ...
        'TerminationMethod','Truncated', ...
        'TracebackDepth',tbl);

    useGPUHere = false;
    if useGPU
        try
            if gpuDeviceCount("available") > 0
                gpuDevice;
                useGPUHere = true;
            end
        catch
            useGPUHere = false;
        end
    end

    snrLin = 10^(snrDb/10);
    Es = 1;
    N0 = Es / snrLin;
    noiseVar = N0/2;

    nBits = 0;
    nErrs = 0;

    while (nBits < maxBits) && (nErrs < minErrs)
        u = randi([0 1], frameBits, 1, 'int8');
        c = enc(u);

        pad = mod(-numel(c), k);
        if pad ~= 0
            c = [c; zeros(pad,1,'like',c)];
        end

        cMat = reshape(c, k, []).';
        symIdx = bi2de(double(cMat), 'left-msb');

        if useGPUHere
            try
                symIdxG = gpuArray(symIdx);
                tx = qammod(symIdxG, M, 'UnitAveragePower', true);

                n = sqrt(noiseVar) .* (randn(size(tx),'like',tx) + 1j*randn(size(tx),'like',tx));
                rx = tx + n;

                llr = qamdemod(rx, M, 'UnitAveragePower', true, ...
                    'OutputType', 'llr', 'NoiseVariance', 2*noiseVar);
                llr = gather(llr);
            catch
                useGPUHere = false;
            end
        end

        if ~useGPUHere
            tx = qammod(symIdx, M, 'UnitAveragePower', true);
            n = sqrt(noiseVar) .* (randn(size(tx)) + 1j*randn(size(tx)));
            rx = tx + n;

            llr = qamdemod(rx, M, 'UnitAveragePower', true, ...
                'OutputType', 'llr', 'NoiseVariance', 2*noiseVar);
        end

        llr = llr(:);
        d = dec(llr);
        d = int8(d(1:frameBits));     % 关键：强制类型一致

        nErrs = nErrs + nnz(u ~= d);
        nBits = nBits + numel(u);
    end

    ber = nErrs / nBits;
end