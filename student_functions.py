import numpy as np
from collections import deque
import heapq

def BFS(matrix, start, end):
    """
    BFS algorithm: Using a queue to store neighbors of the current node,
        visit all the neighbors of the current node before moving to the next depth level.
        
    Parameters:
    ---------------------------
    matrix: np array 
        The graph's adjacency matrix
    start: integer
        starting node
    end: integer
        ending node
    
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes, each key is a visited node,
        each value is the adjacent node visited before it.
    path: list
        Founded path
    """
    # TODO: 
   
    path=[]
    visited={}
    
    # declare a queue
    q = deque([(start, None)])
    
    # loop until the queue is empty
    while q:
        
        # take the first element from the queue
        node, parent = q.popleft()
        
        # if the node is already visited, ignore it
        if node in visited:
            continue
        
        # mark the node as visited
        visited[node] = parent
        
        # if the node is the end node, break the loop
        if node == end:
            break
        
        # traversal from right to left
        for i in range(len(matrix[node]) - 1, -1, -1):
            
            # add neighbors to the queue if have not been visited
            if matrix[node][i] > 0 and i not in visited:
                q.append((i, node))
    
    path = construct_path(visited, end)
                
    return visited, path

def DFS(matrix, start, end):
    """
    DFS algorithm: Using a stack to store neighbors of the current node,
        as stack work similarly to recursion, it will visit the last neighbor of the current node first,
        so that it will visit to the deepest node before moving to next neighbor.
        
    Parameters:
    ---------------------------
    matrix: np array 
        The graph's adjacency matrix
    start: integer 
        starting node
    end: integer
        ending node
    
    Returns
    ---------------------
    visited 
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """

    # TODO: 
    
    path=[]
    visited={}
    # declare a stack
    st = [(start, None)]
    
    # loop until the stack is empty
    while st:
        
        # take the top element from the stack
        node, parent = st.pop()
        
        # if the node is already visited, ignore it
        if node in visited:
            continue
        
        # mark the node as visited
        visited[node] = parent
        
        # if the node is the end node, break the loop
        if node == end:
            break
        
        # traversal from right to left
        for i in range(len(matrix[node])):
            
            # add neighbors to the stack if have not been visited
            if matrix[node][i] > 0 and i not in visited:
                st.append((i, node))
                
    path = construct_path(visited, end)
   
    return visited, path


def UCS(matrix, start, end):
    """
    Uniform Cost Search algorithm: Using a priority queue to store neighbors of the current node,
        visit the neighbor with the smallest cost first, then update the cost of the neighbor 
        if it has a lower cost, and continue to visit the neighbor with the smallest cost.
        
    Parameters:visited
    ---------------------------
    matrix: np array
        The graph's adjacency matrix
    start: integer
        starting node
    end: integer
        ending node
    
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """
    # TODO:  
    path=[]
    visited={}
    
    # declare a priority queue
    pq = [(0, start, None)]
    
    # declare a dictionary to store the minimum cost to reach each node
    min_cost = {start: 0}
    
    # loop until the priority queue is empty
    while pq:
        
        # take the smallest cost element from the priority queue
        cost, node, parent = heapq.heappop(pq)
        
        # if the node is already visited, ignore it
        if node in visited:
            continue
        
        # mark the node as visited
        visited[node] = parent
        
        # if the node is the end node, break the loop
        if node == end:
            break
        
        # traversal from right to left
        for i in range(len(matrix[node]) - 1, -1, -1):
            
            # access neighbors that have not been visited
            if matrix[node][i] > 0 and i not in visited:
                
                # calculate the new cost to reach the neighbor
                new_cost = cost + matrix[node][i]
                
                # if the neighbor has not been reached before or the new cost is smaller than the current cost, 
                # update the cost and add the neighbor to the priority queue
                if i not in min_cost or new_cost < min_cost[i]:
                    min_cost[i] = new_cost
                    heapq.heappush(pq, (new_cost, i, node))

    path = construct_path(visited, end)
    
    return visited, path


def GBFS(matrix, start, end):
    """
    Greedy Best First Search algorithm: Use a priority queue to store neighbors of the current node,
        visit the neighbor with the smallest heuristic value first (edge weight in this case).
         
    heuristic : edge weights
    Parameters:
    ---------------------------
    matrix: np array 
        The graph's adjacency matrix
    start: integer 
        starting node
    end: integer
        ending node
   
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """
    # TODO: 
    path=[]
    visited={}
    
    # declare a priority queue
    pq = [(0, start, None)]

    # loop until the priority queue is empty
    while pq:
        # take the smallest heuristic value element from the priority queue
        _, node, parent = heapq.heappop(pq)

        # if the node is already visited, ignore it
        if node in visited:
            continue
        
        # mark the node as visited
        visited[node] = parent
        
        # if the node is the end node, break the loop
        if node == end:
            break
        
        # traversal from right to left
        for i in range(len(matrix[node]) - 1, -1, -1):
            
            # add neighbors to the priority queue if have not been visited
            if matrix[node][i] > 0 and i not in visited:
                h = matrix[node][i] # h = edge weight
                heapq.heappush(pq, (h, i, node))

    path = construct_path(visited, end)
    
    return visited, path

def Astar(matrix, start, end, pos):
    """
    A* Search algorithm: A combination of UCS and GBFS, it uses path cost (g) and heuristic value (h) to find the
        optimal path. The algorithm uses a priority queue to store neighbors of the current node, visit the neighbor
        with the smallest f value (f = g + h) first and update the cost of the neighbor if it has a lower cost.
    
    heuristic: eclid distance based positions parameter
    Parameters:
    ---------------------------
    matrix: np array UCS
        The graph's adjacency matrix
    start: integer 
        starting node
    end: integer
        ending node
    pos: dictionary. keys are nodes, values are positions
        positions of graph nodes
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """
    # TODO:

    path=[]
    visited={}
    
    # declare a priority queue to store the f value, g value, node, and parent node
    pq = [(0, 0, start, None)]
    
    # declare a dictionary to store the cost to reach each node
    g_score = {start: 0}
    
    # loop until the priority queue is empty
    while pq:
        
        # take the smallest f value element from the priority queue
        _, g, node, parent = heapq.heappop(pq)
        
        # if the node is already visited, ignore it
        if node in visited:
            continue
        
        # mark the node as visited
        visited[node] = parent
        
        # if the node is the end node, break the loop
        if node == end:
            break
        
        # traversal from right to left
        for i in range(len(matrix[node]) - 1, -1, -1):
            
            # access neighbor nodes
            if matrix[node][i] > 0:
                
                # calculate the new cost to reach the neighbor
                new_g = g + matrix[node][i]
                
                # if the neighbor has not been reached before or if the new cost is lower, update the cost
                if i not in g_score or new_g < g_score[i]:
                    # store the updated cost to reach the neighbor
                    g_score[i] = new_g
                    
                    # calculate heuristic value from the neighbor to the end node
                    h = euclidean_distance(pos[i], pos[end]) # h = euclidean distance
                    
                    # calculate the f value
                    f = new_g + h
                    
                    # add the neighbor to the priority queue
                    heapq.heappush(pq, (f, new_g, i, node))

    path = construct_path(visited, end)
    
    return visited, path

def euclidean_distance(p1, p2):
    """
    Euclidean distance between two points
    Parameters:
    ---------------------------
    p1: tuple
        point 1
    p2: tuple
        point 2
    Returns
    ---------------------
    float
        Euclidean distance
    """
    # TODO:
    
    # calculate the Euclidean distance between two points using the formula sqrt((x1-x2)^2 + (y1-y2)^2)
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5

def construct_path(visited, end):
    """
    Construct path from visited nodes
    Parameters:
    ---------------------------
    visited: dictionary
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    end: integer
        ending node
    Returns
    ---------------------
    list
        Founded path
    """
    
    path = []
    
    # if the end node is visited, construct the path
    if end in visited:
        
        # loop to add nodes to the path
        while end is not None:
            path.append(end)
            end = visited[end]
        
        # reverse the path to get the correct order
        path.reverse()
        
    return path