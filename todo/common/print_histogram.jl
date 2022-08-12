export print_histogram

# This is essentially Base.show(io::IO, ::MIME"text/plain", t::Trial)
# in BenchmarkTools.jl

function bindata(sorteddata, nbins, min, max)
    Δ = (max - min) / nbins
    bins = zeros(nbins)
    lastpos = 0
    for i in 1:nbins
        pos = searchsortedlast(sorteddata, min + i * Δ)
        bins[i] = pos - lastpos
        lastpos = pos
    end
    return bins
end

function asciihist(bins, height = 1)
    histbars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
    if minimum(bins) == 0
        barheights = 2 .+
                     round.(Int, (height * length(histbars) - 2) * bins ./ maximum(bins))
        barheights[bins .== 0] .= 1
    else
        barheights = 1 .+
                     round.(Int, (height * length(histbars) - 1) * bins ./ maximum(bins))
    end
    heightmatrix = [min(length(histbars), barheights[b] - (h - 1) * length(histbars))
                    for h in height:-1:1, b in 1:length(bins)]
    return map(height -> if height < 1
                   ' '
               else
                   histbars[height]
               end, heightmatrix)
end

function print_histogram(v::Vector{T}) where {T <: AbstractFloat}
    medv = median(v)
    avgv = mean(v)
    stdv = std(v)
    minv = minimum(v)
    maxv = maximum(v)

    medstr = string(@sprintf("%.3f", medv))
    avgstr = string(@sprintf("%.3f", avgv))
    stdstr = string(@sprintf("%.3f", stdv))
    minstr = string(@sprintf("%.3f", minv))
    maxstr = string(@sprintf("%.3f", maxv))

    lmaxwidth = maximum(length.((medstr, avgstr, minstr)))
    rmaxwidth = maximum(length.((stdstr, maxstr)))

    print(" Range ")
    printstyled("("; color = :light_black)
    printstyled("min"; color = :cyan, bold = true)
    print(" … ")
    printstyled("max"; color = :magenta)
    printstyled("):  "; color = :light_black)
    printstyled(lpad(minstr, lmaxwidth); color = :cyan, bold = true)
    print(" … ")
    printstyled(lpad(maxstr, rmaxwidth); color = :magenta)
    print("  ")
    printstyled("┊"; color = :light_black)

    print("\n Value ")
    printstyled("("; color = :light_black)
    printstyled("median"; color = :blue, bold = true)
    printstyled("):     "; color = :light_black)
    printstyled(lpad(medstr, lmaxwidth), rpad(" ", rmaxwidth + 5); color = :blue,
                bold = true)
    printstyled("┊"; color = :light_black)

    print("\n Value ")
    printstyled("("; color = :light_black)
    printstyled("mean"; color = :green, bold = true)
    print(" ± ")
    printstyled("σ"; color = :green)
    printstyled("):   "; color = :light_black)
    printstyled(lpad(avgstr, lmaxwidth); color = :green, bold = true)
    print(" ± ")
    printstyled(lpad(stdstr, rmaxwidth); color = :green)
    print("  ")
    printstyled("┊"; color = :light_black)

    # The height and width of the printed histogram in characters.
    histheight = 8
    histwidth = 42 + lmaxwidth + rmaxwidth

    histv = sort(v)

    bins = bindata(histv, histwidth - 1, minv, maxv)
    # if median size of (bins with >10% average data/bin) is less than 5% of max 
    # bin size, log the bin sizes
    if median(filter(b -> b > 0.1 * length(v) / histwidth, bins)) / maximum(bins) < 0.05
        bins, logbins = log.(1 .+ bins), true
    else
        logbins = false
    end
    hist = asciihist(bins, histheight)
    hist[:, end - 1] .= ' '
    maxbin = maximum(bins)

    delta1 = (maxv - minv) / (histwidth - 1)
    if delta1 > 0
        medpos = 1 + round(Int, (histv[length(v) ÷ 2] - minv) / delta1)
        avgpos = 1 + round(Int, (avgv - minv) / delta1)
    else
        medpos, avgpos = 1, 1
    end

    print("\n")
    for r in axes(hist, 1)
        print("\n  ")
        for (i, bar) in enumerate(view(hist, r, :))
            color = :default
            if i == avgpos
                color = :green
            end
            if i == medpos
                color = :blue
            end
            printstyled(bar; color = color)
        end
    end

    print("\n  ", minstr)
    caption = "Histogram: " * (logbins ? "log(frequency)" : "frequency")
    if logbins
        printstyled(" "^((histwidth - length(caption)) ÷ 2 - length(minstr));
                    color = :light_black)
        printstyled("Histogram: "; color = :light_black)
        printstyled("log("; bold = true, color = :light_black)
        printstyled("frequency"; color = :light_black)
        printstyled(")"; bold = true, color = :light_black)
    else
        printstyled(" "^((histwidth - length(caption)) ÷ 2 - length(minstr)), caption;
                    color = :light_black)
    end
    print(lpad(maxstr, ceil(Int, (histwidth - length(caption)) / 2) - 1), " ")
    printstyled("<"; bold = true)
    print("\n")
    return nothing
end
