# Reasoning with Sampling: Your Base Model is Smarter Than You Think (arXiv:2510.14901)

Ce papier défend l’idée qu’on peut obtenir une grande partie des gains des modèles “reasoning” post-entraînés (RL) **sans ré-entraîner** le modèle, en utilisant une stratégie d’inférence basée sur un échantillonnage plus intelligent (“Power Sampling”). 

## Contexte

Les modèles de langage génèrent des réponses par échantillonnage token par token. Pour améliorer les performances sur des tâches de raisonnement (maths, code, science), une approche standard est le post-entraînement par renforcement (RL) : générer des sorties, évaluer celles qui réussissent, et pousser le modèle à produire plus souvent ces sorties. 

Le papier examine l’hypothèse que le RL améliore parfois surtout la performance en **repondérant** des sorties déjà présentes dans le modèle de base (augmentation de la probabilité de bonnes trajectoires), plutôt qu’en ajoutant de nouvelles capacités. 

## Objectif : échantillonner dans une distribution “accentuée”

Au lieu d’échantillonner selon la distribution standard :

$$
p(\text{réponse})
$$

la méthode cible une distribution de type :

$$
p(\text{réponse})^\alpha \quad \text{avec } \alpha > 1
$$

Ce “power” renforce la masse de probabilité des séquences globalement plus plausibles et diminue celle des séquences moins plausibles. 

### Différence avec la température

Le papier souligne que ce n’est pas équivalent à baisser la température, car :

- la température agit localement (sur le prochain token),
- la distribution $$p^\alpha$$ agit globalement (sur la séquence complète).

Ils montrent que ces deux opérations induisent des distributions différentes et peuvent privilégier des choix distincts au cours de la génération. 

## Méthode : Metropolis–Hastings (MCMC) pour générer selon $$p^\alpha$$

Échantillonner directement $$p^\alpha$$ est difficile car la normalisation sur l’espace de toutes les séquences est intractable. Les auteurs utilisent **Metropolis–Hastings (MCMC)** pour construire une chaîne dont la distribution stationnaire correspond à $$p^\alpha$$. 

### Principe

1. partir d’une séquence courante,
2. proposer une nouvelle séquence candidate,
3. accepter/rejeter la candidate selon une règle d’acceptation Metropolis–Hastings,
4. répéter plusieurs itérations, puis retourner la séquence finale.

### Adaptation autoregressive

Plutôt que de resampler toute la séquence, la proposition consiste typiquement à :

- choisir une position dans la séquence,
- resampler un suffixe (ou un bloc) via une distribution de proposition,
- appliquer le ratio d’acceptation MH.

Ils décrivent un schéma **Autoregressive MCMC** (Algorithme 1) et discutent l’usage de distributions intermédiaires pour améliorer le mélange de la chaîne. 

## Évaluation

Benchmarks (single-shot) :

- **MATH500**   
- **HumanEval**   
- **GPQA Diamond**   
- **AlpacaEval 2.0**   

Modèles : **Qwen2.5-Math-7B**, **Qwen2.5-7B**, **Phi-3.5-mini-instruct**.  
Baseline RL : **GRPO** (post-training sur MATH). 

## Résultats (points saillants)

La méthode **Power Sampling** :

- améliore nettement le modèle de base,
- dépasse souvent une baseline “low-temperature”,
- se rapproche de GRPO et peut le dépasser selon le modèle/la tâche. 

Extraits (Table 1) :

- **Qwen2.5-Math-7B**
  - MATH500 : 0.496 → 0.748 (GRPO 0.785)
  - HumanEval : 0.329 → 0.573 (GRPO 0.537)
  - GPQA : 0.278 → 0.389 (GRPO 0.399) 

- **Qwen2.5-7B**
  - HumanEval : 0.329 → 0.622 (GRPO 0.561) 

- **Phi-3.5-mini-instruct**
  - HumanEval : 0.213 → 0.732 (GRPO 0.134) 

## Analyse

- **Concentration vs RL** : la procédure rapproche la génération de régions à haute vraisemblance, tandis que GRPO peut être encore plus concentré.   
- **Longueur des solutions** : sur MATH500, la longueur moyenne des solutions peut devenir comparable à celle obtenue via GRPO, sans objectif explicite “écris plus long”.   
- **Diversité (pass@k)** : GRPO tend à plafonner plus tôt (diversité plus faible), tandis que Power Sampling garde une meilleure progression du pass@k pour de grands k, suggérant une meilleure diversité tout en conservant de bonnes performances single-shot.   
- **Hyperparamètres** : performances dépendantes de $$\alpha$$ et du nombre d’itérations MCMC ; un $$\alpha$$ intermédiaire et quelques étapes suffisent souvent avant stabilisation.   

## Implications et limites

- **Test-time scaling** : transfert d’une partie des gains du RL vers du calcul à l’inférence (plusieurs propositions/acceptations internes).   
- **Coût d’inférence** : amélioration contre un coût computationnel plus élevé au moment de répondre ; le papier discute ce compromis et le compare à l’ordre de grandeur d’un entraînement GRPO.   
- **Portée** : ne démontre pas que le RL est inutile ; montre qu’une part importante des gains peut être obtenue sans entraînement additionnel via un schéma d’échantillonnage adapté. 
