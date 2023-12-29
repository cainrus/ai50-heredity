import csv
import itertools
import sys
import numpy as np

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

        "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():
    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):
                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def genes_passed(target_genes_count):
    mutation = PROBS['mutation']
    if target_genes_count == 2:
        # count 2
        return 1 - mutation
    elif target_genes_count == 1:
        # count 1
        return .5 * (1 - mutation)
    else:
        # count 0
        return mutation


def check_parents(mother, father):
    if mother is None and father is not None:
        raise NotImplementedError
    if mother is not None and father is None:
        raise NotImplementedError
    return mother is not None


def get_genes_count(name, one_gene, two_genes):
    return 1 if name in one_gene else 2 if name in two_genes else 0


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    probabilities = []
    genes_per_name = {}
    # Calc parent genes
    for name in people:
        genes_per_name[name] = get_genes_count(name, one_gene, two_genes)

    for name in people:
        person = people[name]
        mother = person['mother']
        father = person['father']
        has_parents = check_parents(mother, father)
        genes_count = get_genes_count(name, one_gene, two_genes)

        # Calculate genes probabilities.
        if has_parents:
            mother_gene_transfer_probability = genes_passed(genes_per_name[mother])
            father_gene_transfer_probability = genes_passed(genes_per_name[father])
            if name in one_gene:
                probabilities.append(
                    mother_gene_transfer_probability * (1 - father_gene_transfer_probability) +
                    father_gene_transfer_probability * (1 - mother_gene_transfer_probability)
                )
            elif name in two_genes:
                probabilities.append(
                    mother_gene_transfer_probability * father_gene_transfer_probability
                )
            else:
                probabilities.append(
                    (1 - mother_gene_transfer_probability) * (1 - father_gene_transfer_probability)
                )
        else:
            probabilities.append(PROBS['gene'][genes_count])

        # Calculate traits probabilities.
        person_trait = person['trait']
        is_expected_trait = name in have_trait # True of False.
        if person_trait is None:
            genes_count = genes_per_name[name] # 0, 1 or 2.
            probabilities.append(PROBS['trait'][genes_count][is_expected_trait])
        elif person_trait is True:
            probabilities.append(1 if is_expected_trait else 0)
        elif person_trait is False:
            probabilities.append(0 if is_expected_trait else 1)

    return np.prod(probabilities)


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for name in probabilities:
        genes_count = get_genes_count(name, one_gene, two_genes)
        probabilities[name]['gene'][genes_count] += p
        trait_present = name in have_trait
        probabilities[name]['trait'][trait_present] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for name in probabilities:
        person_data = probabilities[name]
        for attribute in person_data:
            distribution = person_data[attribute]
            total_probability = sum(distribution.values())
            if total_probability > 0:
                for value_type in distribution:
                    distribution[value_type] /= total_probability


if __name__ == "__main__":
    main()
