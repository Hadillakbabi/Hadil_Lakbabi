import matplotlib.pyplot as plt
from game import create_board
from algorithms import minimax, alphabeta, mcts


def benchmark(board, algorithm, **kwargs):
    """Exécute un algorithme et récupère ses résultats."""
    fonctions = {"minimax": minimax, "alphabeta": alphabeta, "mcts": mcts}
    resultat = fonctions[algorithm](board, **kwargs)

    return {
        "algorithm": algorithm,
        "move": resultat["move"],
        "score": resultat["score"],
        "nodes_explored": resultat["nodes_explored"],
        "time": resultat["time"],
    }


def main():
    board = create_board()
    profondeurs = [1, 2, 3, 4, 5]
    simulations = [100, 500, 1000, 2000]

    resultats_minimax = []
    resultats_alphabeta = []
    resultats_mcts = []

    print(f"{'Algorithm':<12} {'Configuration':<18} {'Move':<6} {'Score':<10} {'Nodes':<10} {'Time (s)'}")
    print("-" * 75)

    for depth in profondeurs:
        for nom, liste in [("minimax", resultats_minimax), ("alphabeta", resultats_alphabeta)]:
            r = benchmark(board, nom, depth=depth)
            liste.append(r)
            print(f"{nom:<12} {'Depth ' + str(depth):<18} {r['move']!s:<6} {r['score']:<10.4f} {r['nodes_explored']:<10} {r['time']:.6f}")

    for n in simulations:
        r = benchmark(board, "mcts", num_simulations=n, seed=42)
        resultats_mcts.append(r)
        print(f"{'mcts':<12} {str(n) + ' simulations':<18} {r['move']!s:<6} {r['score']:<10.4f} {r['nodes_explored']:<10} {r['time']:.6f}")

    # Vérification: les scores doivent être identiques et Alpha-Beta explore moins (ou autant).
    for mini, ab in zip(resultats_minimax, resultats_alphabeta):
        assert mini["score"] == ab["score"], "Scores Minimax / Alpha-Beta différents"
        assert ab["nodes_explored"] <= mini["nodes_explored"], "Alpha-Beta explore trop de nœuds"
    print("\nVérification Minimax / Alpha-Beta : OK")

    # Graphique 1: nombre de nœuds selon la profondeur.
    plt.figure()
    plt.plot(profondeurs, [r["nodes_explored"] for r in resultats_minimax], marker="o", label="Minimax")
    plt.plot(profondeurs, [r["nodes_explored"] for r in resultats_alphabeta], marker="o", label="Alpha-Beta")
    plt.xlabel("Profondeur")
    plt.ylabel("Nombre de nœuds explorés")
    plt.title("Minimax vs Alpha-Beta")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparaison_noeuds.png")

    # Graphique 2: temps selon le nombre de simulations MCTS.
    plt.figure()
    plt.plot(simulations, [r["time"] for r in resultats_mcts], marker="o", label="MCTS")
    plt.xlabel("Nombre de simulations")
    plt.ylabel("Temps d'exécution (s)")
    plt.title("Temps d'exécution de MCTS")
    plt.legend()
    plt.grid(True)
    plt.savefig("temps_mcts.png")

    plt.show()


if __name__ == "__main__":
    main()
