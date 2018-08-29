# Social Network Analysis using Girvann Newman Algorithm

Version:

```html
Spark – 2.2.1, Python – 2.7
```

Betweenness code calculates the betweenness between every two edges in the graph and can be run using the command:

```html
spark-submit betweenness.py [input_file]
```

Community code creates communities in the social graph using the null model and it's modularity `Q(G,S)`. To run this file, execute:

```html
spark-submit community.py [input_file]
```
