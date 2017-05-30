#############################################################################################################
#
#       Date        |       Author          |       Description
#   ----------------|-----------------------|---------------------------------------------------------------
#   May 10, 2017    |   Anurag Dixit        |   Initial Draft
#   May 11, 2017    |   Pavan Joshi         |   Implementing functionalities to add factors to the model
#   May 12, 2017    |   Pavan Joshi         |   Changed Inference method to BeliefPropagation
#   ----------------|-----------------------|---------------------------------------------------------------
#
#############################################################################################################


import os
import numpy as np


from pgmpy.models import MarkovModel
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
from dataprocessor import DataProcessor
from pgmpy.inference import Mplp
import scipy.stats

class MarkovNetwork(object):


        def __init__(self, directoryPath):
    		self.dp = DataProcessor(directoryPath)
		self.model= MarkovModel()

        def addFactor(self,column_one,column_two=None):
            factor = self.dp.getFactors(column_one,column_two)
            if column_two != None:
                variables = [column_one,column_two]
            else:
                variables = [column_one]
            discreteFactor = DiscreteFactor(variables,cardinality=factor['cardinality'],
                                            values=factor['factors'].tolist())
            self.model.add_factors(discreteFactor)


        def define_markov_structure(self):

                print("Constructing the Markov Network Model graph")
                self.model.add_edge('pageCategory','pagePopularity')
                self.model.add_edge('pagePopularity','pageTalkingAbt')
                self.model.add_edge('pageTalkingAbt','Comments')
                self.model.add_edge('postPromotion','Comments')
                self.model.add_edge('postLength','postShareCt')
                self.model.add_edge('postLength','Comments')
                self.model.add_edge('postShareCt','Comments')
                self.model.add_edge('baseDay','cc2')
                self.model.add_edge('cc1','cc2')
                self.model.add_edge('cc2','cc3')
                self.model.add_edge('cc3','Comments')
                self.model.add_edge('pageCheckins','Comments')
                self.model.add_edge('postDay','cc4')
                self.model.add_edge('cc4','Comments')
                print("===============================================================")
                print("Adding Factors to the graph")
                self.addFactor('pageCategory')
                self.addFactor('pagePopularity')
                self.addFactor('pageTalkingAbt')
                self.addFactor('Comments')
                self.addFactor('postPromotion')
                self.addFactor('postShareCt')
                self.addFactor('postLength')
                self.addFactor('baseDay')
                self.addFactor('cc1')
                self.addFactor('cc2')
                self.addFactor('cc3')
                self.addFactor('pageCheckins')
                self.addFactor('postDay')
                self.addFactor('cc4')

                self.addFactor('pageCategory','pagePopularity')
                self.addFactor('pagePopularity','pageTalkingAbt')
                self.addFactor('pageTalkingAbt','Comments')
                self.addFactor('postPromotion','Comments')
                self.addFactor('postLength','postShareCt')
                self.addFactor('postLength','Comments')
                self.addFactor('postShareCt','Comments')
                self.addFactor('baseDay','cc2')
                self.addFactor('cc1','cc2')
                self.addFactor('cc2','cc3')
                self.addFactor('cc3','Comments')
                self.addFactor('pageCheckins','Comments')
                self.addFactor('postDay','cc4')
                self.addFactor('cc4','Comments')
                print("===============================================================")
                self.mplp = Mplp(self.model)

	def get_independencies(self):
		#return self.model.get_local_independecies()
		pass

	def probability_query(self, query, evidences=None):

		inference = BeliefPropagation(self.model)
		#Assume that the samples are generated in the form of numpy matrix

		if(evidences == None):
			#Need to return the normalized result of the query
	            res = inference.query(query.keys())

		else:
	            res = inference.query(query.keys(),evidences)
			#For all the evidences extract the values
			#"""for key in evidences.keys():
                        #        val = self.dp.getBin(evidences[key])
                        #        tmp = gen[np.ix_(gen[:,key] > val, (self.dp.idx['Comments']))]
                        #        res = tmp.sum()/gen[:,self.dp.idx['Comments']].sum()
			#"""
		return res


	def infer(self):
		f = open('query.txt', 'r')
                lines = f.readlines()
                for i in lines:
                        lst = i.strip().split(", ")
                        queryType = lst[0]
                        if(queryType == 'I'):
                                print "\n\n##########    Printing all independencies    ##########\n"
                                print self.model.get_local_independencies()
                        elif(queryType == 'CP'):
       				evidences = dict()
      				query = dict()
            			args = lst[1].split("=")
	                        key = args[0].strip().strip(" ")
            			query[key] = int(self.dp.getBin(int(args[1].strip().strip(" ")),args[0].strip()))
            			args = lst[2].split(" -> ")
            			for arg in args[1].strip("[").strip("]").split("&"):
            				evid = arg.strip().strip(" ").split("=")
            				evidences[evid[0].strip().strip(" ")] = int(self.dp.getBin(int(evid[1].strip().strip(" ")),evid[0].strip()))
            			print "\n\n##########    P(",lst[1],"|",lst[2].split(" -> ")[1],")    ##########\n"
            			print self.probability_query(query,evidences)[key].values[query[key]]
                        elif(queryType == 'M'):
                                query = dict()
                                args = lst[1].split("=")
                                query[args[0].strip().strip(" ")] = int(self.dp.getBin(int(args[1].strip().strip(" ")),args[0].strip()))
                                if args[0].strip() not in self.dp.discrete:
                                    print "\n\n##########    P(",args[0],"<",args[1],")    ##########\n"
                                else:
                                    print "\n\n##########    P(",lst[1],")    ##########\n"
                                print self.probability_query(query)[args[0].strip()].values[query[args[0].strip()]]
                        elif(queryType == 'PF'):
                                #For partition function of the model graph
                                print "\n\n##########    Partition Function    #############\n"
                                print self.model.get_partition_function()
                        elif(queryType == 'MB'):
                                #For markov blanket
				print "\n\n##########    Markov Blankets   #############\n"
                                print "Markov Blanket of ", lst[1],": ",self.model.markov_blanket(lst[1])
                        elif(queryType == 'MQ'):
                                #MAP Query
                                print "\n\n##########    MAP query using MPLP   #############\n"
                                result = self.mplp.map_query()
                                print result
                        elif(queryType == 'FT'):
                                #Finding Triangles
                                print "\n\n##########    Triangles present in the model   #############\n"
                                print "Triangles: ",self.mplp.find_triangles()
                        elif(queryType == 'IG'):
                                #Integrality GAP, the lower the exact solution
                                print "\n\n##########    Integrality gap   #############\n"
                                print "Integrality gap: "+str(self.mplp.get_integrality_gap())

                #TODO: Add handling of multiple types of queries defined in query file

        def metric(self, a, b):

                #Make sure the input parameters are probability Distributions

                entropy = scipy.stats.entropy(a)
                kl_divergence = scipy.stats.entropy(a, b)
                return entropy, kl_divergence


	def evaluation_metrics(self):

		#gibbs = GibbsSampling(self.model)
                #gen = gibbs.sample(size=1000)
		#samples = np.array(gen[:,self.dp.idx['Comments']])
                
		samples = self.probability_query({'Comments':None})['Comments'].values
		origin_data = self.dp.getFactors('Comments')['factors']
                #originalpdf = scipy.stats.norm(np.mean(origin_data),np.var(origin_data))
		#query_domain = np.linspace(np.mean(origin_data) - np.var(origin_data),
                #                                                np.mean(origin_data) + np.var(origin_data), 100)
		mean = self.dp.metadata['Comments']['mean']
		#print len(samples)
		#print len(origin_data)
                #TODO: Need to add new code
                entropy, kl_divergence = self.metric(samples,origin_data)
                print "\n\n##########    Performance Evaluation Metrics    ##########\n"
                print "Mean:",mean
                print "Entropy:",entropy
                print "KL Divergence:",kl_divergence


if  __name__=="__main__":
	directoryPath = os.path.abspath("Training/")
	mm = MarkovNetwork(directoryPath)
	mm.define_markov_structure()
	mm.infer()
	mm.evaluation_metrics()
