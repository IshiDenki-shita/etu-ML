
# The ideas for character segmentation

## core idea

Japanese character can be reconstructed by certain parts.
Each parts has common shape like cross, L shape or etc.
Skeletonizing probably clarify the shapes.
And then we can split chars with graph analytics

## association of ideas

noise removement{
    detect long line with hough transformation
    recognize the line as noise
}

making graph{
    skeltonize
    detect cross and end-point
    connect relating points
}

detect char roughly{
    contour detection
}

## the flow

1. remove noise
2. skeltonize
3. line detection
4. remove the too long line as a noise
5. 