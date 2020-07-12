import numpy as np

class BayesianNetwork:
    def __init__(self, filename):
        self.bayesNet = {}
        self.bayesNetForApprox = {}
        f = open(filename, 'r') 
        N = int(f.readline())
        lines = f.readlines()
        for line in lines:
            node, parents, domain, shape, probabilities = self.__extract_model(line)
            # YOUR CODE HERE
            if (parents != []):
                domainParent = []
                for parent in parents:
                    parentNode = self.bayesNet[parent]
                    domainParent.append(parentNode.domain)

                self.bayesNet[node] = Node(node, parents, domain, probabilities, domainParent)
                self.bayesNetForApprox[node] = Node(node, parents, domain, probabilities, domainParent)
            else:
                self.bayesNet[node] = Node(node, parents, domain, probabilities)
                self.bayesNetForApprox[node] = Node(node, parents, domain, probabilities)
        f.close()

    def exact_inference(self, filename):
        result = 0
        f = open(filename, 'r')
        query_variables, evidence_variables = self.__extract_query(f.readline())
        # YOUR CODE HERE
        # query_variables, evidence_variables is dictionary {}

        if (len(evidence_variables) == 0):          # without evidence
            
            # Reduce var not in query_variable
            while (not self.isStop(query_variables)):
                chooseNode, amount, listNode = self.chooseNode(query_variables)
                # print("Choose Node:", chooseNode)
                # print("ListNode:", listNode)

                if (amount == 1):
                    self.reduceBySum(listNode[0], chooseNode)
                else:
                    while(len(listNode) > 1):            
                        nodeName1 = listNode[0]
                        nodeName2 = listNode[1]
                        self.mulMatrix(nodeName1, nodeName2, chooseNode)
                        self.bayesNet.pop(listNode[0])
                        listNode.pop(0)

                    # remove variable by sum
                    nodeName = listNode[0]
                    node = self.bayesNet[nodeName]
                    node.probabilities = np.sum(node.probabilities, node.getIndex(chooseNode))
                    node.updateSize(node.probabilities.size)    #Update size after sum
                    node.removeDomain(chooseNode)               #Remove domain of variable
                    node.removeRemainVar(chooseNode)            #Remove variable from remainVars

                # for node in self.bayesNet.values():
                #     node.print()
                # print("-------------------------------------")
            processed_variable = []
            # Multiply matrix with variable remaining
            while len(self.bayesNet.keys()) > 1:
                chooseNode, amount, listNode = self.chooseNode(processed_variable)
                if amount == 1:
                    processed_variable.append(chooseNode)
                else:
                    processed_variable.append(chooseNode)
                    while len(listNode) > 1:
                        nodeName1 = listNode[0]
                        nodeName2 = listNode[1]
                        self.mulMatrix(nodeName1, nodeName2, chooseNode)
                        self.bayesNet.pop(listNode[0])
                        listNode.pop(0)
            # Remain only 1 node -> Result
            nodeName, node = self.bayesNet.popitem()
            for i in range(len(node.remainVar)):
                var = node.remainVar[i]
                value = query_variables[var]
                index = node.domainAll[i].index(value)
                node.probabilities = node.probabilities.take(index, 0)
            result = node.probabilities
        else:                                       #with evidence

            #Reduce evidence_variables
            for node in self.bayesNet.values():
                node.reduceElement(evidence_variables)
                # node.print()

            
            #Reduce variables not in query_variable
            query_vars = list(query_variables.keys())
            evidence_vars = list(evidence_variables.keys())
            while (not self.isStop(query_vars + evidence_vars)):
                chooseNode, amount, listNode = self.chooseNode(query_vars + evidence_vars)

                # print("ChooseNode:",chooseNode)
                # print("Amount:", amount)
                # print("ListNode:", listNode)

                if (amount == 1):
                    self.reduceBySum(listNode[0], chooseNode)
                else:
                    while(len(listNode) > 1):            
                        nodeName1 = listNode[0]
                        nodeName2 = listNode[1]
                        self.mulMatrix(nodeName1, nodeName2, chooseNode)
                        self.bayesNet.pop(listNode[0])
                        listNode.pop(0)

                    # remove variable by sum
                    nodeName = listNode[0]
                    node = self.bayesNet[nodeName]
                    node.probabilities = np.sum(node.probabilities, node.getIndex(chooseNode))
                    node.updateSize(node.probabilities.size)    #Update size after sum
                    node.removeDomain(chooseNode)               #Remove domain of variable
                    node.removeRemainVar(chooseNode)            #Remove variable from remainVar

                # print("-------------------------------------------")

            #Multiply matrix with variable remaining

            processed_variable = []
            while len(self.bayesNet.keys()) > 1:
                chooseNode, amount, listNode = self.chooseNode(processed_variable)
                

                if (amount == 1):
                    processed_variable.append(chooseNode)
                else:
                    processed_variable.append(chooseNode)
                    while len(listNode) > 1:
                        nodeName1 = listNode[0]
                        nodeName2 = listNode[1]
                        self.mulMatrix(nodeName1, nodeName2, chooseNode)
                        self.bayesNet.pop(listNode[0])
                        listNode.pop(0)

            #Remain only 1 node -> result
            nodeName, node = self.bayesNet.popitem()
            probabilitiesTable = node.probabilities.copy()
            value = None
            for i in range(len(node.remainVar)):
                var = node.remainVar[i]
                if (var in query_variables):
                    value = query_variables[var]
                else:
                    value = evidence_variables[var]
                index = node.domainAll[i].index(value)
                node.probabilities = node.probabilities.take(index, 0)

            total = probabilitiesTable.sum()
            result = node.probabilities / total
        f.close()
        return result

    def approx_inference(self, filename):
        result = 0
        f = open(filename, 'r')
        # YOUR CODE HERE
        query_variable, evidence_variables = self.__extract_query(f.readline())
        
        allVar = []
        domainAll = []
        # for un_evidence
        sampleAmount = 200000
        validSample = 0
        # for evidence
        sumLikeLiHood = 0
        validLikeLiHood = 0
    
        # Transform table probability to uniform distribution table
        for node in self.bayesNetForApprox.values():
            if (node.nodeName not in evidence_variables):
                allVar.append(node.nodeName)
                domainAll.append(node.domain)
                probabilitiesTable = np.reshape(node.probabilities, (node.probabilities.size,))
                size = len(node.domainAll[-1])

                tmp = 0
                for i in range(probabilitiesTable.size):
                    if (i % size == 0):
                        tmp = probabilitiesTable[i]
                    else:
                        probabilitiesTable[i] += tmp
                        tmp = probabilitiesTable[i]
            
                node.probabilities = np.reshape(probabilitiesTable, node.probabilities.shape)
        
        # Create sample
        flagEvidence = len(evidence_variables) > 0      # If have evidence_variables
        lengthAllVar = len(allVar)
        for _ in range(sampleAmount):
            varCreated = {}
            randomValue = np.random.uniform(0,1,lengthAllVar)
            for i in range(lengthAllVar):
                var = allVar[i]
                node = self.bayesNetForApprox[var]
                tableProbability = node.probabilities

                if len(node.remainVar) > 1:     # Node with parent
                    for parentVar in node.remainVar[:-1]:
                        if (parentVar in varCreated):   # Var in allVar (not in evidence)
                            index = varCreated[parentVar]
                        else:                           # Var in evidence
                            node = self.bayesNetForApprox[parentVar]
                            value = evidence_variables[parentVar]
                            index = node.domain.index(value)

                        tableProbability = tableProbability.take(index, 0)
            
                    for index in range(tableProbability.size):
                        value = tableProbability[index]
                        if (randomValue[i] < value):
                            varCreated[var] = index
                            break
                else:                           # Node without parent (1 attribute)
                    for index in range(node.probabilities.size):
                        value = node.probabilities[index]
                        if (randomValue[i] < value):
                            varCreated[var] = index
                            break
        
            if not flagEvidence:
                flag = True
                for var in query_variable:
                    value = query_variable[var]
                    index = self.bayesNetForApprox[var].domain.index(value)
                    if (index != varCreated[var]):
                        flag = False
                        break
                if (flag):
                    validSample += 1
            else:
                likelihood = 1
                for var in evidence_variables:
                    tableProbability = self.bayesNetForApprox[var].probabilities
                    value = evidence_variables[var]
                    index = self.bayesNetForApprox[var].domain.index(value)
                    
                    if len(self.bayesNetForApprox[var].remainVar) > 1:  # have parent
                        for parentVar in node.remainVar[:-1]:
                            index = varCreated[parentVar]
                            tableProbability = tableProbability.take(index, 0)
            
                        likelihood *= tableProbability[index]
                    else:                                               # have not parent
                        likelihood *= tableProbability[index]

                sumLikeLiHood += likelihood
                flag = True
                for var in query_variable:
                    value = query_variable[var]
                    index = self.bayesNetForApprox[var].domain.index(value)
                    if index != varCreated[var]:
                        flag = False
                        break
                if (flag):
                    validLikeLiHood += likelihood


        if flagEvidence:
            result = validLikeLiHood / sumLikeLiHood
        else:
            result = validSample / sampleAmount

        f.close()
        return result

    def __extract_model(self, line):
        parts = line.split(';')
        node = parts[0]
        if parts[1] == '':
            parents = []
        else:
            parents = parts[1].split(',')
        domain = parts[2].split(',')
        shape = eval(parts[3])
        probabilities = np.array(eval(parts[4])).reshape(shape)
        return node, parents, domain, shape, probabilities

    def __extract_query(self, line):
        parts = line.split(';')

        # extract query variables
        query_variables = {}
        for item in parts[0].split(','):
            if item is None or item == '':
                continue
            lst = item.split('=')
            query_variables[lst[0]] = lst[1]

        # extract evidence variables
        evidence_variables = {}
        for item in parts[1].split(','):
            if item is None or item == '':
                continue
            lst = item.split('=')
            evidence_variables[lst[0]] = lst[1]
        return query_variables, evidence_variables

    def isStop(self, query_variables = {}):
        for node in self.bayesNet.values():
            if (node.remainVar != []):
                for var in node.remainVar:
                    if var not in query_variables:
                        return False
        return True

    def chooseNode(self, query_variables = {}):
        countList = {}
        for node in self.bayesNet.values():
            for var in node.remainVar:
                if (var not in query_variables):
                    if (var in countList):
                        countList[var] += 1
                    else:
                        countList[var] = 1

        # Return Var with minimum number of occurrences and list of Node have it.
        chooseNode, min = countList.popitem()
        listNode = []
        listVar = []
        for var in countList:
            if (countList[var] < min):
                min = countList[var]
                chooseNode = var

        # Sort variable by size
        for node in self.bayesNet.values():
            if (chooseNode in node.remainVar):
                listNode.append(node)
        listNode.sort(key=lambda x: x.size)

        for node in listNode:
            listVar.append(node.nodeName)
        
        return chooseNode, min, listVar

    def mulMatrix(self, nodeName1, nodeName2, chooseNode):
        node1 = self.bayesNet[nodeName1]
        node2 = self.bayesNet[nodeName2]
        proba1= node1.probabilities
        proba2 = node2.probabilities

        query = self.createQuery(nodeName1, nodeName2, chooseNode)
        node2.probabilities = np.einsum(query, proba1, proba2)
        node2.size = node2.probabilities.size

    def createQuery(self, nodeName1, nodeName2, chooseNode):
        node1 = self.bayesNet[nodeName1]
        node2 = self.bayesNet[nodeName2]

        queryForNode1 = ""      # It will like abcdef
        queryForNode2 = ""

        intChar = 97            # character a
        for _ in range(len(node1.remainVar)):
            queryForNode1 += chr(intChar)
            intChar += 1

        remainVarOfNode1 = node1.remainVar
        for var in node2.remainVar:
            if (var in remainVarOfNode1):
                index = node1.remainVar.index(var)
                queryForNode2 += queryForNode1[index]
            else:
                queryForNode2 += chr(intChar)
                intChar += 1
        indexChooseNode = node1.remainVar.index(chooseNode)
        #slice char at indexChooseNode
        remainVariable = queryForNode1[:indexChooseNode] + queryForNode1[indexChooseNode+1:]
        queryForResult = queryForNode2

        listCharAppend = []
        for char in remainVariable:
            if (char not in queryForNode2):
                queryForResult += char
                listCharAppend.append(char)

        for char in listCharAppend:
            index = ord(char) - 97
            var = node1.remainVar[index]
            node2.domainAll.append(node1.getDomain(var))
            node2.allVar.append(node1.remainVar[index])
            node2.remainVar.append(node1.remainVar[index])

        return queryForNode1 + ',' + queryForNode2 + "->" + queryForResult

    def reduceBySum(self, nodeName, var):
        node = self.bayesNet[nodeName]
        index = node.getIndex(var)
        node.probabilities = np.sum(node.probabilities, index)
        node.removeDomain(var)
        node.removeRemainVar(var)
        node.updateSize(node.probabilities.size)


class Node:
    def __init__(self, nodeName, parents, domain, probabilities, domainParent = []): # parent = [], domain = [], probabilities = []
        self.nodeName = nodeName
        self.domain = domain                       #domain: domain of nodeName
        self.allVar = parents + [nodeName]         #All of Variable in Node
        self.remainVar = parents + [nodeName]          #Variable need to remove
        self.probabilities = probabilities          #Probabilities Table
        self.domainAll = domainParent +[domain]    #All of Variable's Domain
        self.size = probabilities.size              #Number of record

    def print(self):
        print("Node Name:",self.nodeName)
        print("Domain:", self.domain)
        print("Domain All:", self.domainAll)
        print("All Variables:", self.allVar)
        print("Remain Variables:", self.remainVar)
        print("Probabilities Table:", self.probabilities)
        print("Size:", self.size)
        
    def getDomain(self, nodeName):      # Return domain of a variable
        index = self.remainVar.index(nodeName)
        return self.domainAll[index].copy()

    def getIndex(self, var):            # Return index of a variable - dimension
        return self.remainVar.index(var)

    def removeRemainVar(self, var):
        self.remainVar.remove(var)

    def removeAllVar(self, var):
        self.allVar.remove(var)

    def removeDomain(self, var):
        index = self.getIndex(var)
        self.domainAll.pop(index)

    def updateSize(self, size):
        self.size = size

    def reduceElement(self, vars):
        reduceVars = [x for x in vars]

        check = []
        for var in self.allVar:
            if (var in reduceVars):
                check.append(True)
            else:
                check.append(False)
        
        shape = []
        for i in range(len(check)):
            if check[i] == True:
                var = self.allVar[i]
                value = vars[var]
                index = self.domainAll[i].index(value)
                self.probabilities = np.array(([self.probabilities.take(index, i)]))
                self.domainAll[i] = [value]
                shape.append(1)
            else:
                shape.append(len(self.domainAll[i]))
        self.probabilities = np.reshape(self.probabilities, tuple(shape))   # Reshape table
        self.size = self.probabilities.size

        # print(self.remainVar)
        # print(self.nodeName)
        # print(self.domainAll)
        # print(self.allVar)
        # print(self.probabilities)

        



        

        
            





        
        