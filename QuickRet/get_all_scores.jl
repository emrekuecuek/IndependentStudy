
using DataStructures, DelimitedFiles, DataFrames, Serialization, Base.Iterators
using LinearAlgebra: diag

include("quickret_phase1_1.jl")

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

template_list = readdir("./interfaces/");
test_list = deserialize("test_list");
quickret_obj, totalfeaturecount = deserialize("quickret_object_and_totalfeaturecount");

function get_matchlist_length_for_each_interface(matchlist)
    matchlist_length = OrderedDict()
    for (key,value) in matchlist
        matchlist_length[key] = length(value)
    end
    return matchlist_length
end

function get_longest_transformation_match_for_each_interface(bestposes)
    transformation_lengths = OrderedDict()
    for (best_key, best_value) in bestposes
        transformation_lengths[best_key] = maximum(length.(values(best_value)))
    end
#     reversesortlist = OrderedDict()
#     for (key,value) in transformation_lengths
#         if !haskey(reversesortlist,value)
#             reversesortlist[value] = []
#         end
#         push!(reversesortlist[value],key)
#     end
#     return sort(reversesortlist, rev=true)
    return transformation_lengths
end

function create_score_list(query_chain, quickret_obj)
    featurebaselist = retrievefeaturebaselist(query_chain);
    matchlist = creatematchlist(featurebaselist,quickret_obj);
    matchlist_lengths = get_matchlist_length_for_each_interface(matchlist);
    bestposes = createbestpose(matchlist);
    transformation_lengths = get_longest_transformation_match_for_each_interface(bestposes);
    return (matchlist_lengths, transformation_lengths)
end

full_score_data = OrderedDict{Tuple, OrderedDict{UInt16, NTuple{2,UInt16}}}()
@time for protein in test_list
    try
        query_protein = retrievepdb(protein[1:4], dir="./test_pdb_files/", remove_disorder=true);
        println(query_protein)
	flush(stdout)
        Threads.@threads for chain_id in protein[5:6]
            matchlist_lengths, transformation_lengths = create_score_list(query_protein[chain_id], quickret_obj)
            
            matchlist_transformation = OrderedDict{UInt16, NTuple{2,UInt16}}()
            for (match_key, match_value) in matchlist_lengths
                matchlist_transformation[match_key] = (match_value, transformation_lengths[match_key]);
            end
            full_score_data[protein[1:4],chain_id] = matchlist_transformation
        end
        catch e
    end
end

using Serialization
serialize("full_score_data", full_score_data)
