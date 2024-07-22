## Privacy Can Arise Endogenously in an Economic System with Learning Agents

This repository contains the code for the paper [Privacy Can Arise Endogenously
in an Economic System with Learning
Agents](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.FORC.2024.9) by Nivasini Ananthakrishnan,
Tiffany Ding, Mariel Werner, Sai Praneeth Karimireddy, and Michael I. Jordan

## Abstract

We study price-discrimination games between buyers and a seller where privacy
arises endogenously - that is, utility maximization yields equilibrium
strategies where privacy occurs naturally. In this game, buyers with a high
valuation for a good have an incentive to keep their valuation private, lest
the seller charge them a higher price. This yields an equilibrium where some
buyers will send a signal that misrepresents their type with some probability;
we refer to this as buyer-induced privacy. When the seller is able to publicly
commit to providing a certain privacy level, we find that their equilibrium
response is to commit to ignore buyers' signals with some positive probability;
we refer to this as seller-induced privacy. We then turn our attention to
a repeated interaction setting where the game parameters are unknown and the
seller cannot credibly commit to a level of seller-induced privacy. In this
setting, players must learn strategies based on information revealed in past
rounds. We find that, even without commitment ability, seller-induced privacy
arises as a result of reputation building. We characterize the resulting
seller-induced privacy and seller’s utility under no-regret and
no-policy-regret learning algorithms and verify these results through
simulations.

## Code Execution

Simply run the following command to reproduce plots in the paper.
```
python3 run_dynamics.py
```
