# Virtual Graph Convolutional Networks (VGCN)

In the virtual graph, a virtual node represents a timepoint. The embedding of a virtual node represents the “infection situations” at the timepoint. The virtual edges connect two virtual nodes at two timepoints, and the edge weights measure the significance of the virtual edge. Since the timepoints connected by the virtual edges can be outside the time window, VGCN can break the limitation of the time window by learning from neighbor nodes and improve the predictive accuracy.

