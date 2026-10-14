# Visual flow and detailed workspace

The default Visual flow follows the model from symbols and embeddings through parallel heads, residual additions, per-position processing, and prediction. Its colors and curved connections draw inspiration from Georgia Tech's [Transformer Explainer](https://poloclub.github.io/transformer-explainer/). The interface and SVG are implemented locally; no reference-site code or assets are embedded.

Clicking a diagram stage selects the same node as the Detailed workspace. Head presets, exact strength, attention redirection, prediction reset, and structured input editing continue to use the existing worker and completed-experiment state. Detailed retains the complete graph, effect traces, source-copy editor, vector inspectors, attention arithmetic, ablation explorer, and residual readouts. Choosing residual copying or a guided trace opens that workspace. Save/replay and runtime controls are available through a disclosure in Visual flow.

Both views share one experiment; switching views does not reload a model, mutate an intervention, or recompute a comparison. The diagram uses completed tokens and results while a new input is still a draft or a comparison is pending. `?view=details` opens the complete workspace directly, including for the existing detailed regression suite.

Main-diagram connections encode computation order and branching, not activation values or causal attribution. Normalization is folded into stages. The optional attention diagram uses the four largest actual attention weights for the completed head and query, labels positions to distinguish repeated symbols, and identifies their percentages. It is omitted for residual-copy experiments, whose display does not supply an attention matrix. Prediction bars show actual probabilities from the same completed comparison as the detailed result panel.

The canvas scrolls within a bounded region on narrow screens. Diagram nodes are keyboard buttons; selected and edited states have text/outline cues in addition to color. Detailed and Visual flow are available without clearing either input drafts or completed results.
