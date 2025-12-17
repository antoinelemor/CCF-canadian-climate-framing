#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROJECT:
-------
CCF-Canadian-Climate-Framing

PAPER:
------
CCF_Methodology

TITLE:
------
9_network_cocitation.py

MAIN OBJECTIVE:
---------------
Build a co-citation network of persons mentioned in Canadian climate coverage
(2024) to identify epistemic authorities and their relationships. The network
reveals which individuals are frequently mentioned together, exposing the
social structure of climate discourse.

Dependencies:
-------------
- pandas
- networkx
- sqlalchemy
- config_db (local module)

MAIN FEATURES:
-------------
1) Load NER entities from all 2024 sentences
2) Extract and normalize person names
3) Aggregate persons at the article level
4) Build co-occurrence network (edge = two persons in same article)
5) Calculate network metrics (degree centrality, communities)
6) Export to GEXF format for Gephi visualization

Author:
-------
Antoine Lemor
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, List, Set
from itertools import combinations

import pandas as pd
import networkx as nx
from sqlalchemy import create_engine, text
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DB_PARAMS: Dict[str, Any] = {
    "host":     os.getenv("CCF_DB_HOST", "localhost"),
    "port":     int(os.getenv("CCF_DB_PORT", 5432)),
    "dbname":   "CCF_Database",
    "user":     os.getenv("CCF_DB_USER", "antoine"),
    "password": os.getenv("CCF_DB_PASS", ""),
    "options":  "-c client_min_messages=warning",
}
TABLE_NAME = "CCF_processed_data"

# Output directories
SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR.parent / "Outputs" / "Figures"
STATS_DIR = SCRIPT_DIR.parent / "Outputs" / "Stats"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)

# Network parameters
TOP_N_NODES = 100  # Keep top N persons by mention frequency
MIN_EDGE_WEIGHT = 3  # Minimum co-occurrences to create an edge


# =============================================================================
# NAME NORMALIZATION
# =============================================================================

# Comprehensive name normalization mapping
NAME_VARIANTS: Dict[str, List[str]] = {
    # Canadian Federal Politicians
    'Justin Trudeau': ['J. Trudeau', 'Prime Minister Trudeau', 'PM Trudeau', 'Trudeau', 'Justin'],
    'Pierre Poilievre': ['P. Poilievre', 'Poilievre'],
    'Jagmeet Singh': ['J. Singh', 'Singh'],
    'Chrystia Freeland': ['C. Freeland', 'Freeland'],
    'Steven Guilbeault': ['S. Guilbeault', 'Guilbeault'],
    'Mark Holland': ['M. Holland', 'Holland'],
    'Jonathan Wilkinson': ['J. Wilkinson', 'Wilkinson'],
    'Marc Garneau': ['M. Garneau', 'Garneau'],
    'Catherine McKenna': ['C. McKenna', 'McKenna', 'Mckenna', 'Catherine Mckenna'],
    'Elizabeth May': ['E. May'],

    # Former Federal Politicians
    'Stephen Harper': ['S. Harper', 'Prime Minister Harper', 'PM Harper', 'Harper'],
    'Andrew Scheer': ['A. Scheer', 'Scheer'],
    'Erin O\'Toole': ['E. O\'Toole', 'O\'Toole', 'Otoole'],
    'Michael Ignatieff': ['M. Ignatieff', 'Ignatieff'],
    'Thomas Mulcair': ['T. Mulcair', 'Mulcair'],
    'Jack Layton': ['J. Layton', 'Layton'],

    # Provincial Premiers
    'Doug Ford': ['D. Ford', 'Premier Ford', 'Ford'],
    'François Legault': ['F. Legault', 'Legault'],
    'Danielle Smith': ['D. Smith', 'Premier Smith'],
    'David Eby': ['D. Eby', 'Eby', 'Premier Eby'],
    'Scott Moe': ['S. Moe', 'Moe'],
    'Heather Stefanson': ['H. Stefanson', 'Stefanson'],
    'Tim Houston': ['T. Houston', 'Houston'],
    'Blaine Higgs': ['B. Higgs', 'Higgs'],
    'Dennis King': ['D. King'],
    'Andrew Furey': ['A. Furey', 'Furey'],

    # Former Premiers
    'Jason Kenney': ['J. Kenney', 'Kenney'],
    'Rachel Notley': ['R. Notley', 'Notley'],
    'John Horgan': ['J. Horgan', 'Horgan'],
    'Christy Clark': ['C. Clark'],
    'Kathleen Wynne': ['K. Wynne', 'Wynne'],
    'Brad Wall': ['B. Wall', 'Wall'],

    # BC Politicians
    'John Rustad': ['J. Rustad', 'Rustad'],
    'Sonia Furstenau': ['S. Furstenau', 'Furstenau'],

    # Alberta Politicians
    'Brian Jean': ['B. Jean', 'Jean'],
    'Tyler Shandro': ['T. Shandro', 'Shandro'],

    # US Politicians
    'Donald Trump': ['D. Trump', 'President Trump', 'Trump'],
    'Joe Biden': ['J. Biden', 'President Biden', 'Biden'],
    'Kamala Harris': ['K. Harris', 'Vice President Harris', 'Harris'],
    'Barack Obama': ['B. Obama', 'President Obama', 'Obama'],
    'Hillary Clinton': ['H. Clinton', 'Clinton'],
    'John Kerry': ['J. Kerry', 'Kerry'],
    'Al Gore': ['A. Gore', 'Gore'],
    'Elon Musk': ['E. Musk', 'Musk'],

    # International Leaders
    'Emmanuel Macron': ['E. Macron', 'Macron'],
    'Boris Johnson': ['B. Johnson'],
    'Rishi Sunak': ['R. Sunak', 'Sunak'],
    'Xi Jinping': ['Xi'],
    'Vladimir Putin': ['V. Putin', 'Putin'],

    # Scientists & Experts
    'David Suzuki': ['D. Suzuki', 'Suzuki'],
    'Mark Carney': ['M. Carney', 'Carney'],
    'Greta Thunberg': ['G. Thunberg', 'Thunberg'],

    # Indigenous Leaders
    'Perry Bellegarde': ['P. Bellegarde', 'Bellegarde'],
    'RoseAnne Archibald': ['R. Archibald', 'Archibald'],
}

# Build reverse lookup
NAME_LOOKUP: Dict[str, str] = {}
for canonical, variants in NAME_VARIANTS.items():
    NAME_LOOKUP[canonical.lower()] = canonical
    for variant in variants:
        NAME_LOOKUP[variant.lower()] = canonical


def normalize_person_name(name: str) -> str:
    """Normalize a person's name to canonical form."""
    if not name or not isinstance(name, str):
        return ""

    # Clean up
    name = ' '.join(name.split()).strip()

    # Handle comma-separated names (keep first)
    if ',' in name:
        name = name.split(',')[0].strip()

    # Handle concatenated names (keep first two words if too long)
    name_parts = name.split()
    if len(name_parts) > 3:
        name = ' '.join(name_parts[:2])

    # Title case
    name = name.title()

    # Check lookup
    name_lower = name.lower()
    if name_lower in NAME_LOOKUP:
        return NAME_LOOKUP[name_lower]

    # Filter out single-word names that are too common/ambiguous
    # (except for known single-name variants already in lookup)
    if len(name_parts) == 1 and len(name) < 6:
        return ""

    return name


def extract_persons_from_ner(ner_value) -> List[str]:
    """Extract person names from NER JSON field."""
    if pd.isna(ner_value) or ner_value in ('', '{}'):
        return []

    try:
        if isinstance(ner_value, str):
            data = json.loads(ner_value)
            if isinstance(data, dict) and 'PER' in data:
                persons = data.get('PER', [])
                normalized = []
                for p in persons:
                    if p and isinstance(p, str):
                        norm = normalize_person_name(p)
                        if norm:
                            normalized.append(norm)
                return normalized
    except (json.JSONDecodeError, TypeError):
        pass

    return []


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data() -> pd.DataFrame:
    """Load 2024 NER data from the CCF database."""
    print("=" * 70)
    print("CO-CITATION NETWORK OF PERSONS IN CLIMATE COVERAGE (2024)")
    print("=" * 70)
    print("\n[1/5] Loading NER data from CCF_Database...")

    engine = create_engine(
        f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}"
        f"@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}",
        connect_args={"options": DB_PARAMS['options']}
    )

    sql = text(f"""
        SELECT doc_id, ner_entities
        FROM "{TABLE_NAME}"
        WHERE EXTRACT(YEAR FROM date) = 2024
          AND ner_entities IS NOT NULL
          AND ner_entities != '{{}}'
    """)

    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    engine.dispose()

    print(f"    → Loaded {len(df):,} sentences with NER data")
    print(f"    → From {df['doc_id'].nunique():,} articles")

    return df


# =============================================================================
# NETWORK CONSTRUCTION
# =============================================================================

def aggregate_persons_by_article(df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Aggregate unique persons mentioned in each article."""
    print("\n[2/5] Aggregating persons by article...")

    # Extract persons for each sentence
    df['persons'] = df['ner_entities'].apply(extract_persons_from_ner)

    # Aggregate unique persons per article
    article_persons: Dict[str, Set[str]] = defaultdict(set)

    for _, row in df.iterrows():
        for person in row['persons']:
            article_persons[row['doc_id']].add(person)

    # Count total persons
    all_persons = set()
    for persons in article_persons.values():
        all_persons.update(persons)

    print(f"    → Found {len(all_persons):,} unique persons")
    print(f"    → Across {len(article_persons):,} articles")

    return article_persons


def count_person_mentions(article_persons: Dict[str, Set[str]]) -> Counter:
    """Count total mentions of each person across articles."""
    person_counts = Counter()
    for persons in article_persons.values():
        for person in persons:
            person_counts[person] += 1
    return person_counts


def build_cooccurrence_network(
    article_persons: Dict[str, Set[str]],
    top_n: int = TOP_N_NODES,
    min_weight: int = MIN_EDGE_WEIGHT
) -> nx.Graph:
    """Build co-occurrence network from article-level data."""
    print("\n[3/5] Building co-occurrence network...")

    # Get top N persons by mention count
    person_counts = count_person_mentions(article_persons)
    top_persons = set([p for p, _ in person_counts.most_common(top_n)])

    print(f"    → Keeping top {len(top_persons)} persons by mention frequency")

    # Count co-occurrences
    cooccurrence: Counter = Counter()

    for persons in article_persons.values():
        # Filter to top persons
        persons_in_article = persons & top_persons

        # Count all pairs
        for p1, p2 in combinations(sorted(persons_in_article), 2):
            cooccurrence[(p1, p2)] += 1

    print(f"    → Found {len(cooccurrence):,} person pairs")

    # Build network
    G = nx.Graph()

    # Add nodes with attributes
    for person in top_persons:
        G.add_node(
            person,
            mentions=person_counts[person],
            label=person
        )

    # Add edges (only if weight >= min_weight)
    edges_added = 0
    for (p1, p2), weight in cooccurrence.items():
        if weight >= min_weight:
            G.add_edge(p1, p2, weight=weight)
            edges_added += 1

    print(f"    → Network has {G.number_of_nodes()} nodes and {edges_added} edges")
    print(f"    → (Edges with weight >= {min_weight})")

    return G


# =============================================================================
# NETWORK ANALYSIS
# =============================================================================

def compute_network_metrics(G: nx.Graph) -> nx.Graph:
    """Compute centrality metrics and community detection."""
    print("\n[4/5] Computing network metrics...")

    # Degree centrality
    degree_cent = nx.degree_centrality(G)
    for node, cent in degree_cent.items():
        G.nodes[node]['degree_centrality'] = round(cent, 4)

    # Weighted degree (sum of edge weights)
    for node in G.nodes():
        weighted_deg = sum(G[node][neighbor]['weight'] for neighbor in G.neighbors(node))
        G.nodes[node]['weighted_degree'] = weighted_deg

    # Betweenness centrality
    between_cent = nx.betweenness_centrality(G, weight='weight')
    for node, cent in between_cent.items():
        G.nodes[node]['betweenness_centrality'] = round(cent, 4)

    # Eigenvector centrality (may fail on disconnected graphs)
    try:
        eigen_cent = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
        for node, cent in eigen_cent.items():
            G.nodes[node]['eigenvector_centrality'] = round(cent, 4)
    except nx.NetworkXError:
        print("    → Warning: Eigenvector centrality failed (disconnected graph)")
        for node in G.nodes():
            G.nodes[node]['eigenvector_centrality'] = 0.0

    # Community detection (Louvain)
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G, weight='weight')
        for node, comm in partition.items():
            G.nodes[node]['community'] = comm
        n_communities = len(set(partition.values()))
        print(f"    → Detected {n_communities} communities (Louvain)")
    except ImportError:
        print("    → Warning: python-louvain not installed, using greedy modularity")
        communities = nx.community.greedy_modularity_communities(G, weight='weight')
        for i, comm in enumerate(communities):
            for node in comm:
                G.nodes[node]['community'] = i
        n_communities = len(communities)
        print(f"    → Detected {n_communities} communities (greedy modularity)")

    # Network-level metrics
    if nx.is_connected(G):
        clustering = nx.average_clustering(G, weight='weight')
        print(f"    → Average clustering coefficient: {clustering:.3f}")
    else:
        print(f"    → Graph has {nx.number_connected_components(G)} connected components")
        largest_cc = max(nx.connected_components(G), key=len)
        G_largest = G.subgraph(largest_cc).copy()
        clustering = nx.average_clustering(G_largest, weight='weight')
        print(f"    → Clustering (largest component): {clustering:.3f}")

    return G


# =============================================================================
# OUTPUT
# =============================================================================

def export_gexf(G: nx.Graph) -> None:
    """Export network to GEXF format for Gephi."""
    print("\n[5/5] Exporting network...")

    # Set node size based on mentions (for Gephi visualization)
    max_mentions = max(G.nodes[n]['mentions'] for n in G.nodes())
    for node in G.nodes():
        G.nodes[node]['size'] = G.nodes[node]['mentions'] / max_mentions * 100

    # Export GEXF
    output_path = FIGURES_DIR / "network_cocitation_2024.gexf"
    nx.write_gexf(G, output_path)
    print(f"    → GEXF saved: {output_path}")

    # Export node statistics
    node_stats = []
    for node in G.nodes():
        node_data = G.nodes[node]
        node_stats.append({
            'person': node,
            'mentions': node_data['mentions'],
            'degree_centrality': node_data['degree_centrality'],
            'betweenness_centrality': node_data['betweenness_centrality'],
            'eigenvector_centrality': node_data.get('eigenvector_centrality', 0),
            'weighted_degree': node_data['weighted_degree'],
            'community': node_data.get('community', -1)
        })

    stats_df = pd.DataFrame(node_stats)
    stats_df = stats_df.sort_values('mentions', ascending=False)

    stats_path = STATS_DIR / "network_node_statistics_2024.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"    → Node statistics saved: {stats_path}")

    # Print top 15 by degree centrality
    print("\n" + "=" * 70)
    print("TOP 15 PERSONS BY DEGREE CENTRALITY")
    print("=" * 70)
    print(f"{'Person':<25} {'Mentions':>10} {'Degree':>10} {'Betweenness':>12}")
    print("-" * 70)

    top_by_degree = stats_df.nlargest(15, 'degree_centrality')
    for row in top_by_degree.itertuples():
        print(f"{row.person:<25} {row.mentions:>10,} {row.degree_centrality:>10.3f} {row.betweenness_centrality:>12.3f}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    """Main execution pipeline."""

    # Load data
    df = load_data()

    # Aggregate persons by article
    article_persons = aggregate_persons_by_article(df)

    # Build network
    G = build_cooccurrence_network(article_persons)

    # Compute metrics
    G = compute_network_metrics(G)

    # Export
    export_gexf(G)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Open network_cocitation_2024.gexf in Gephi")
    print("2. Apply layout (e.g., ForceAtlas2)")
    print("3. Size nodes by 'mentions' or 'degree_centrality'")
    print("4. Color nodes by 'community'")
    print("5. Export as PDF/PNG")


if __name__ == "__main__":
    main()
