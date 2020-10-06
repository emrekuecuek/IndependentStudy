
using DataStructures, DelimitedFiles, DataFrames, Serialization, Base.Iterators, Random
using LinearAlgebra: diag

Random.seed!(1);

include("quickret_phase1_1.jl")

# One built with only using atoms is inside previous versions.
function featureextractor(base::Array{Float16,2})
    dihedral = rad2deg(dihedralangle((base[:,3] - base[:,4]), (base[:,2] - base[:,3]), (base[:,1] - base[:,2])))
    alpha = rad2deg(bondangle((base[:,1] - base[:,2]), (base[:,3] - base[:,2])))
    beta = rad2deg(bondangle((base[:,2] - base[:,3]), (base[:,4] - base[:,3])))
    dist = √(sum((base[:,2] .- base[:,3]).^2))
    return (alpha, beta, dihedral, dist)
end

function featureextractor(c1_b::Atom, c1_a::Atom, c2_a::Atom, c2_b::Atom)
    dihedral = rad2deg(dihedralangle(c2_b, c2_a, c1_a, c1_b))
    alpha = rad2deg(bondangle(c1_b, c1_a, c2_a))
    beta = rad2deg(bondangle(c1_a, c2_a, c2_b))
    dist = distance(c1_a, c2_a)
    return (alpha, beta, dihedral, dist) 
end

# Previous version can be found on older commits.
function discretizer(features, table_type::String; feature_thresholds::NTuple{4,Float64} = quickret_obj.feature_thresholds)::Tuple
    if table_type == "matchlist"
        return tuple([Int8(features[i] ÷ feature_thresholds[i]) for i in 1:length(features)]...)
    elseif table_type == "posetable"
        return tuple([Int16(features[i] ÷ 1) for i in 1:length(features)]...)
    else
        throw(ErrorException("Wrong table type entered"))
    end
end

function retrievefeaturebaselist(protein)
    featurebaselist = []
    alphaatoms = collectatoms(protein, calphaselector);
    betaatoms = Dict(zip(resnumber.(collectatoms(protein, cbetaselector)), collectatoms(protein, cbetaselector)));
    for (first_key, i) in enumerate(alphaatoms)
        for (second_key, j) in enumerate(alphaatoms)
            dist = distance(i, j)
            if 4<dist<13
                i_b = betaatoms[resnumber(i)]
                j_b = betaatoms[resnumber(j)]
                if i_b.name == " CB " && j_b.name == " CB "
                    base = Float16.(hcat(i_b.coords, i.coords, j.coords, j_b.coords))
#                     features = featureextractor(base)
                    features = featureextractor(i_b, i, j, j_b)
#                     println(featureextractor(base))
#                     println(features)
                    push!(featurebaselist, (features, base))                 
                end
            end
        end
    end
    return featurebaselist;
    
end

function creatematchlist(featurebaselist, quickret_obj)
    matchlist = Dict{UInt16, Array{NTuple{2, Array{Float16, 2}}}}()
    for (feature, protein_base) in featurebaselist
        try
            for (interface_base, interface_id) in quickret_obj.interface_hashtable[discretizer(feature,"matchlist")]
                if !haskey(matchlist,interface_id)
                    matchlist[interface_id] = []
                end
                push!(matchlist[interface_id], (protein_base, interface_base))
            end
        catch e
#             println(e)
        end
    end
    return matchlist
end

# Creates a list with matching scores. TODO: change function name more properly.
# Also move the match score part into another function. Only calculate number of matches here.
# Calculate score inside a scoring function.
function createsortlist(matchlist,totalfeaturecount)
    sortlist = OrderedDict()
    for (key,value) in matchlist
        sortlist[key] = length(value)/totalfeaturecount[key]
    end
    return sortlist
end

function createbestpose(matchlist)
    bestposes = OrderedDict{UInt16, Dict{NTuple{2, NTuple{3, Int16}}, Array{NTuple{2,Array{Float16,2}}}}}()
    for (key,value) in matchlist
        posetable = Dict{NTuple{2, NTuple{3, Int16}}, Array{NTuple{2,Array{Float16,2}}}}()
        for (int_base, query_base) in value
            temp_transformation =Transformation(int_base, query_base)
            posetable_key = (Tuple(discretizer(temp_transformation.trans1-temp_transformation.trans2,"posetable")), Tuple(discretizer(diag(temp_transformation.rot), "posetable")))
            if !haskey(posetable, posetable_key)
                posetable[posetable_key] = []
            end
            push!(posetable[posetable_key], (int_base, query_base))
        end
        bestposes[key] = posetable
    end
    return bestposes
end

# Move matching score here. Change createsortlist function as match counter only.
function calculatescore(sortlist, bestposes)
    for (best_key, best_value) in bestposes
        sortlist[best_key] += (maximum(length.(values(best_value))))^2/sum(length.(values(best_value)))
    end
    reversesortlist = OrderedDict()
    for (key,value) in sortlist
        if !haskey(reversesortlist,value)
            reversesortlist[value] = []
        end
        push!(reversesortlist[value],key)
    end
    return sort(reversesortlist, rev=true)
end

# Clusters from PIFACE.
piface_data = readdlm("./finalInterfaceClusters_2013_January_24_.txt", '\t');
clusters = [];
for i in 1:size(piface_data, 1)
    temp_clusters = piface_data[i,8:end];
    temp_clusters = temp_clusters[temp_clusters.!=""];
    push!(clusters, temp_clusters);
end
result_df = DataFrame(piface_data[:,[2,4,6]]);
result_df."x4" = clusters;

template_list = readdir("./interfaces/");
cluster_member_list = result_df[in.(result_df.x3, Ref(unique([uppercase(i[1:6]) for i in template_list]))), :];
test_list = rand.(cluster_member_list[length.(cluster_member_list.x4) .>= 5, 4]);

# template_list = readdir("./interfaces/");
# quickret_obj = QuickRet(feature_thresholds = (5.0, 5.0, 5.0, 1.0), transformation_thresholds = Nothing);
# @time totalfeaturecount = hash_template(quickret_obj, hash_list);
quickret_obj, totalfeaturecount = deserialize("quickret_object_and_totalfeaturecount");

function chain_calculator(pdb_id, chain_list, a, template_list, quickret_obj, totalfeaturecount)
    query_protein = retrievepdb(pdb_id, dir="./test_pdb_files/", remove_disorder=true);
    total_chain_success = 0
    total_chains = 0
        
    Threads.@threads for i in chain_list
        
        featurebaselist = retrievefeaturebaselist(query_protein[i]);
        matchlist = creatematchlist(featurebaselist,quickret_obj);
        sortlist = createsortlist(matchlist, totalfeaturecount);
        bestpose = createbestpose(matchlist);
        scores = calculatescore(sortlist,bestpose);
        result = 0
        if length(scores) != 0
            for i in take(values(scores),5)
                if length(result_df[result_df.x3 .== uppercase(template_list[i][1][1:6]), 4]) != 0
                    result += sum(occursin.(query_protein.name[1:4],result_df[result_df.x3 .== uppercase(template_list[i][1][1:6]), 4][1]))
                end
            end
            if result > 0
                total_chain_success += 1
            end
            total_chains += 1
        end
    end
    println(pdb_id,chain_list, " => ($(total_chain_success),$(total_chains))",)
    flush(stdout)
end

Threads.@threads for protein in test_list
    try
        chain_calculator(protein[1:4], protein[5:6], result_df, template_list, quickret_obj, totalfeaturecount)
    catch e
    end
end
