using BioStructures

"""

    QuickRet(;feature_thresholds::NTuple{4, Float64}, transformation_thresholds)

Constructs a QuickRet object. Contains `interface_hashtable`, `feature_thresholds` and `transformation_thresholds`.

QuickRet is a hashing algorithm for protein interfaces. Main point is to determine which cluster a certain given protein belongs to.

A protein consists of `atoms`. When certain atoms interacts and bonds to each other, they constitue a `residue`. A sequence of residues are called as a `chain`. A protein can contain one or more chains.

There are thousands of experimentally determined structures of proteins. Some of them contains a protein itself, some contain different proteins interacting. These interactions are similar in various ways, and in literature there are studies that clustered all interactions in order to prevent redundancy [10.1371/journal.pone.0086738](https://doi.org/10.1371/journal.pone.0086738).

In [PIFACE](https://doi.org/10.1371/journal.pone.0086738), there are clusters in order to prevent redundancy amongst protein interfaces. All of Protein-Protein Interactions (PPI) are clustered, thus redundancy is prevented. Each cluster has a `representative`, that represents the members of its cluster. A useful idea might be to check for similarities for an arbitrary protein with these clusters. Let's call this arbitrary protein as `query protein`. That way, we can determine the specifications of a query protein and investigate its properties, possible interaction candidates. This way, we can computationally calculate its binding properties or possible strong interactions much accurate.

Computational methods are by far faster, and knowing a certain proteins interaction candidates may lead more precise experimental results.

Moreover, cruciality of computational methods are undeniable. Template-based PPI Prediction methods are proven to be successful [1](https://doi.org/10.1093/nar/gku397), [2](https://doi.org/10.1038/nprot.2011.367). Though experimental methods are the most reliable methodologies for obtaining new information on a certain structure, computational methods can lead way to experimental methods since they can be done faster and produce bulk results. However, computational methods have a long way, they can be more faster, accurate.

`QuickRet` aims to decrease the calculation time of PPI's and remaining the accuracy same. As discussed above, template-based PPI prediction algorithms are successful. Template-based method means that to predict an interaction between two arbitrary protein ``P_a`` and ``P_b``, we can use another interaction let's say between ``P_c`` and ``P_d`` as a reference. By that, PRISM tries to dock ``P_a`` and ``P_b`` each other as if they are interacting similar like ``P_c`` and ``P_d``. But for an arbitrary interaction, it takes quite some time to calculate energy scores for each template. Considering the numbers of templates, finding the correct template takes a lot of time. QuickRet tries to reduce time to find possible templates by discarding the most irrelevant templates in a fast manner.

It contains two different steps. In the first step, algorithm deposits a representative interface from each non redundant protein clusters to a hash table. Clusters are obtained from [PIFACE [10.1371]](https://doi.org/10.1371/journal.pone.0086738), a non-redundant template data base for protein protein interactions.

After constructing a hash table from the representatives of all clusters that taken from PIFACE, we can determine what a query protein is structurally similar to hashed clusters by looking the hash table. QuickRet is a fast algorithm, and the reason behind is it uses hash tables and compare proteins by the most little possible parts.

But first let's explain first phase of the algorithm. In the first step, algorithm takes representatives from each cluster. Then, it extracts a small part of it called `base`. A base consists of 4 atoms, 2 Carbon Alpha atoms (CA) and 2 Carbon Beta atoms (CB). CA atoms obtained from different residues between distance of 4 Angstrom to 13 Angstrom. CB atoms are obtained from those residues.

For a better understanding, let's call ``CA_1`` and ``CB_1`` as CA and CB atoms from first residue, and ``CA_2`` and ``CB_2`` for second residue. Algorithm creates a ``base`` from these four atoms, then calculates four `features`. These are:

- Distance between ``CA_1`` and ``CA_2``.
- Bond angle between ``CB_1``, ``CA_1``, ``CA_2``.
- Bond angle between ``CA_1``, ``CA_2``, ``CB_2``.
- Dihedral angle by all atoms.

These features will be `key`s for hash table with one addition. By using the `feature_thresholds` in the object, we will make integer division amongst all features. Hence, all features will be discretized. Then an item of `value` of the `interface_hashtable` will be a tuple, as (`base`, `interface id`) where `base` is a matrix with dimensions ``3x4``, each column is the coordinate of an atom going as ``CB_1``, ``CA_1``, ``CA_2``, ``CB_2``; and `interface id` is the id of the interface which `base` is extracted to. There will be many `base`s with same features (or keys) so all `base`s will create many lists under same `key`s. These can be from different interfaces or same interfaces, it does not matter.

This concludes first phase, hence the main cause and functionality of QuickRet structure. Calculation and insertion of hash table happens in `hash_template`. To reach second phase, see `create_score_list`.

# Arguments
- `interface_hashtable=Dict{NTuple{4,Int64}, Array{Tuple{Array{Float16, 2},Int64}}}()` : A hash table constructured from an interface list.  
- `feature_thresholds::NTuple{4, Float64}` : A tuple of thresholds for inserting bases to the interface_hashtable.
- `transformation_thresholds` : Transformation thresholds.
"""
struct QuickRet
    interface_hashtable::Dict{NTuple{4,Int8}, Array{Tuple{Array{Float16, 2},UInt16}}}
#     interface_hashtable::Dict{NTuple{4,Int64}, Array{Tuple{Array{Array{Float64, 1}, 1},Int64}}}
    feature_thresholds::NTuple{4, Float64}
    transformation_thresholds
end

function QuickRet(;feature_thresholds::NTuple{4, Float64}, transformation_thresholds)::QuickRet
    interface_hashtable = Dict{NTuple{4,Int64}, Array{Tuple{Array{Float16, 2},Int64}}}()
    return QuickRet(interface_hashtable, feature_thresholds, transformation_thresholds)
    
end

# This function discretizes given bases. TODO: update query part. It is not being used, maybe discard that part?
function (q::QuickRet)(features::NTuple{4, Float64}, protein_type::String)::Tuple
    if protein_type == "template"
        return tuple([Int8(features[i] ÷ q.feature_thresholds[i]) for i in 1:length(features)]...)
    elseif protein_type == "query"
        return tuple([features[i] ÷ q.transformation_thresholds[i] for i in 1:length(features)]...)
    else
        throw(ErrorException("Wrong protein type entered"))
    end
end


# In this function, features are calculated. I think it's pretty straightforward. Readers can check
# BioStructures.jl documentation for related functions. Returns a tuple.
function (q::QuickRet)(base::Array{Atom, 1})::Tuple #NTuple{4, Int64}
    dihedral = rad2deg(dihedralangle(base[4], base[3], base[2], base[1]))
    alpha = rad2deg(bondangle(base[1], base[2], base[3]))
    beta = rad2deg(bondangle(base[2], base[3], base[4]))
    dist = distance(base[2], base[3])
    return (alpha, beta, dihedral, dist)
end


# This function retrieves carbon beta atoms because interfaces are only have carbon alpha atoms.
# Takes a protein structure as first parameter, this part is a bit sloppy, normally all it requires is a name (string) for 
# PDB file. Second parameter is the chain, we need it because we don't want all structure's CB atoms, we only need the 
# related chain.
function (q::QuickRet)(interface::ProteinStructure, chain::Char)
#     We are creating a dictionary because we don't need all of the beta atoms, and it is easy to reach them by their
#     residue number. Keys are residue numbers as string, values are atoms.
    betaatoms = Dict{String, Atom}()
#     We are retrieving the full pdb file because we need beta atoms.
    full_pdb = read("./pdb_files/" * uppercase(interface.name[1:4]) * ".pdb", PDB, remove_disorder=true)
#     We only need one chain, so...
    for i in resids(interface[chain]);
#     We are only taking non-glycine CB atoms, because they return CA atoms anyway. Also algorithm does not includes 
#     Glycine atoms,so that's another reason why we extract them.
        full_pdb[chain][i].name != "GLY" ? betaatoms[i] = full_pdb[chain][i].atoms[" CB "] : betaatoms[i] = full_pdb[chain][i].atoms[" CA "]
    end
    return betaatoms
end

# Takes a protein structure and interface id. Extracts bases and calculates features. After that, inserts it into the
# QuickRet object. Hash table structure is given as below:

# Hash Table:
# 
# (5, 3, 7, 7) => {(base, 7), (base, 12), (base, 33453)}
# (35, 66, 12, 1) => {(base, 1), (base, 22)}
# (44, 65, 2, 4) => {(base, 66)}
#     .
#     .
#     .
# 
# Each key represents a list. These lists have tuple inside them containing a base and interface id.
# It is trivial that if you discretize each base with feature_thresholds you will obtain the key of hash table.
# And interface id keeps us in track which base is it belongs to. So in future, if we look at a query protein, 
# when we extract the features of it, we will obtain the list of bases for each feature. And in theory, there will be
# many different interfaces with same features. So by logic, we can say that if we have a match between a query protein's 
# base's discretized features and hash table's key, we are finding small similarities between them. So this is important for us 
# to calculate.
function (q::QuickRet)(interface::ProteinStructure, id::UInt16)
#     Obtaining CA and CB atoms. Beta atoms are extracted differently because our dataset of interfaces only contains CA atoms.
    alphaatoms = collectatoms(interface, calphaselector);
#     To understand how these are extracted, go the function's documentation. This function is a bit old so don't judge.
#     TODO: Change beta atom function so that it takes interface name as parameter rather than all protein structure.
    betaatoms = q(interface, interface.name[8]);
#     We will keep the number of bases for each interface in order to normalize the scoring. Some interfaces might be long,
#     some might be short, so this value is vital.
    totalbase = 0;
#     Calculating everything for each interface. According to algorithm, we need every CA-CB pair that satisfies the distance
#     condition. (4<d(CA_1,CA_2)<13).
    for i in alphaatoms
        for j in alphaatoms
            dist = distance(i,j)
            if 4<dist<13
#     Getting CB atoms. betaatoms has key value pairs with (ResidueId) => (Related CB Atom).
                i_b = betaatoms[resid(i)]
                j_b = betaatoms[resid(j)]
                if i_b.name == " CB " && j_b.name == " CB "
#     Creating a temporary base. This is from Atoms, it is easier to calculate accurate features, so we store them as 4 atoms
#     in the beginning.
                    base = Array{Atom, 1}()
                    push!(base, i_b, i, j, j_b)
#     This is the part where features are calculated. See the related function for documentation. It is a tuple with 
#     first bond angle, second bond angle, dihedral angle, CA distance, respectively.
                    features = q(base)
#     We are discretizing here. Built as same tuple but values are discretized by using feature_thresholds.
                    key = q(features, "template")
#     This part we are creating the base that stored. It is a 3x4 matrix, each column is an atom. CB_1, CA_1, CA_2, CB_2,
#     respectively.
                    base = Float16.(hcat(i_b.coords, i.coords, j.coords, j_b.coords))
#     This part if we calculated these features before, we will push new base at the end of the existing list. If not, we
#     are creating an empty array, then pushing the new base to it.
                    haskey(q.interface_hashtable, key) ? push!(q.interface_hashtable[key], (base, id)) : (q.interface_hashtable[key] = []; push!(q.interface_hashtable[key], (base, id)))
#     Incrementing number of totalbase.
                    totalbase = totalbase + 1
                end
            end
        end
    end
#     Hash table will already be inserted for the instance interface. No need to do anything more about it.
#     Return total number of bases.
#     TODO: Add another field to QuickRet as totalbase. Then those values can be inserted there, so no need for extra
#     work. Would save some trouble.
    return totalbase;
end



"""
    hash_template(q::QuickRet, _list::Array{String, 1})

Takes two arguments, a QuickRet object and an interface list. Reads interfaces from list, makes all necessary calculations, puts them into the hash table, returns the total number of bases for each interfaces.
"""
function hash_template(q::QuickRet, _list::Array{String, 1})
#     TODO: Change interface reading part. It should take the directory as parameter. Or reduce the number of parameters by giving list of files instead of file names.
#     int_base will hold number of total calculated bases for each interface.
    int_base = Dict{UInt16,Int}()
#     Some interfaces might not work due to problems in original PDB file or interface file. So we are using try-catch to
#     discard non-working interfaces. We still need most of it statistically, so if problematic ones are not much, 
#     it should not create a problem
    for (id,int) in enumerate(_list)
        try
#     We are reading interface file here. using BioStructure's read function, file type as PDB, and we are discarding
#     disordered atoms. TODO: Add mmCIF module for hashing. In second part we are calculating features and putting them
#     into the hash table.
            interface = read("./interfaces/" * int, PDB, remove_disorder=true);
            int_base[UInt16(id)] = quickret_obj(interface, UInt16(id));
        catch e
#     Printing errors, and keep calculating.
            println(id, "=>", int, ", Error: ", e)
        end
    end
    return int_base
end