
# idea for character segmentation

## core idea

I imagine the curve splitting characters in cell.
The curve go between chars as possible as it keep far from chars.
But some curve will stack in connected chars
So the maze solving algorytm will be effective

## keyword

- watershed
- A*/Dijkstra
- DP

## imaginated flow

1. make distance map
2. calc gradient for each pixels
3. find valley lines
4. connect the valley lines to top and bottom
5. solve maze problem for stagnant lines in deadend
6. list up the valley lines as border candidates
7. find char features for DP
8. run DP