from Metrics import *
import time

from Node import Node


class DecisionTree():
    def __init__(self, criterion='gini'):
        if criterion == 'entropy':
            self.criterion = entropy_gain
        elif criterion == "entropy_ratio":
            self.criterion = gain_ratio_entropy
        elif criterion == "gini_ratio":
            self.criterion = gain_ratio_gini
        elif criterion == "gini":
            self.criterion = gini_gain
        else:
            self.criterion = gini_gain

        self.attribute_names = None
        self.attribute_classes = []  # different classes in every attribute
        self.prediction_classes = []
        self.used_attributes = []
        self.x = None
        self.y = None
        self.used_attributes = None
        self.root = None
        self.max_time = 60
        self.init_time = -1

    def rec_fit(self, node, x, y, available_attribute_index):
        # if all predictions are the same we have the decision
        if np.unique(y).shape[0] == 1:
            node.predictions[y[0]] = 1
            return

        # if we have different possible predictions then we use probabilities
        for p_class in np.unique(y):
            node.predictions[p_class] = y[(p_class == y)].shape[0]/y.shape[0]

        # if max time is reached then fit is ended
        if self.max_time <= time.time() - self.init_time:
            return

        # no more attributes to keep training
        if x.shape[0] == 0 or x.shape[1] == 0:
            return

        # calculate the best attribute
        evaluation_metrics = []
        for i in range(x.shape[1]):
            evaluation_metrics.append(self.criterion(x[:, i], y, self.attribute_classes[i], self.prediction_classes))
        attribute_index = np.argmax(np.array(evaluation_metrics))
        node.set_attribute_index(available_attribute_index[attribute_index])

        # create child nodes
        for a_class in np.unique(x[:, attribute_index]):
            prediction_classes = np.unique(y[x[:, attribute_index] == a_class])
            node.add_child((a_class, Node(node, prediction_classes)))

        # call recursive fit with new x and y
        indices = np.arange(x.shape[1])
        valid_cols = indices != attribute_index
        for child_i in range(len(node.childs)):
            valid_rows = x[:, attribute_index] == node.childs[child_i][0]
            new_x = x[valid_rows, :]
            new_x = new_x[:, valid_cols]
            new_y = y[valid_rows]
            new_available_index = available_attribute_index[valid_cols]
            self.rec_fit(node.childs[child_i][1], new_x, new_y, new_available_index)

        return

    def fit(self, x, y, attribute_names=None):
        self.init_time = time.time()

        # creating or adding attribute names
        if attribute_names is not None and x.shape[1] == len(attribute_names):
            self.attribute_names = attribute_names
        else:
            self.attribute_names = []
            for j in range(x.shape[1]):
                self.attribute_names.append("Attribute"+str(j))
        self.x = x
        self.y = y
        self.prediction_classes = np.unique(y)
        available_index = np.array(range(x.shape[1]))
        self.root = Node(None, self.prediction_classes, "Root")
        for j in range(x.shape[1]):
            self.attribute_classes.append(np.unique(x[:, j]))
        self.rec_fit(self.root, self.x, self.y, available_index)

    def rec_predict(self, x, node):
        t = x[node.attribute_index]
        # a = self.attribute_names[node.attribute_index]
        # print(str(a)+" -> "+str(t))
        if len(node.childs) == 0:
            # print("Predict: " + str(node.predictions))
            return max(node.predictions, key=node.predictions.get)
        
        next_node = False
        
        for child in node.childs:
            if child[0] == t:
                next_node = child[1]
                break

        if next_node == False:
            return max(node.predictions, key=node.predictions.get)

        return self.rec_predict(x, next_node)

    def predict(self, x):
        if len(x.shape) > 1:
            predictions = np.zeros(x.shape[0], dtype=object)
            for i in range(x.shape[0]):
                predictions[i] = (self.rec_predict(x[i, :], self.root))
            return predictions
        else:
            return self.rec_predict(x, self.root)

    def prune(self):
        pass

    def rec_str(self, out, node, level):
        level += 1
        if len(node.childs) == 0:
            out += " ==> "
            max_key = max(node.predictions, key=node.predictions.get)
            out += max_key

        for child in node.childs:
            out += "\n"
            out += (level*"--------")
            out += "|" + self.attribute_names[node.attribute_index] + " --> " + str(child[0])
            out = self.rec_str(out, child[1], level)
        level -= 1
        return out

    def __str__(self):
        out = "\n"
        out += "Root"
        return self.rec_str(out, self.root, 0)





