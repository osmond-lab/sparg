# covid

Using COVID dataset from GISAID to test sparg. This contains 72 sample sequences (plus a reference) with geographic information about the samples. Samples are aligned and passed into ARGweaver to build the ARG. This is then used as the input into sparg.

# data

 - reference.fasta (downloaded https://gisaid.org/wiv04/ on April 10, 2023, originally named EPI_ISL_402124.fasta): reference sequence for COVID
 - Europe_20230320-20230410
    - original (download from https://www.epicov.org/epi3/frontend#44999b on April 10, 2023): 578 COVID sequences from across Europe uploaded between March 20 and April 10, 2023. Filtered for Complete, High coverage, and Collection date complete.
        - dates_and_locations.tsv
        - nucleotide_sequences.fasta
        - sequencing_technology_metadata.tsv
        - sequencing_technology_metadata_with_latlon.tsv
    - filtered
        - nucleotide_sequences_filtered_aligned.fasta
        - nucleotide_sequences_filtered_aligned.fasta
        - sequencing_technology_metadata_filtered.tsv
