#%%
import sys

sys.path.append('../../')
from components.RAG.generate_retrieval_metrics import retrieval_accuracy

# #%%
# # Generate BLS Accuracy
#
# validation_questions_fp = '../../../data/retrieval_results/bls_qa_questions.xlsx'
# retrieval_fp = '../../../data/retrieval_results/bls_retrieval.xlsx'
#
# retrieval_accuracy(validation_questions_fp=validation_questions_fp, retrieval_fp=retrieval_fp,
#                    comparison_fp='../../../data/retrieval_results/accuracy_retrieval_bls.xlsx')

#%%
# Generate IPEDs Accuracy
validation_questions_fp = '../../../data/retrieval_results/ipeds_qa_questions.xlsx'
retrieval_fp = '../../../data/retrieval_results/ipeds_retrieval.xlsx'

retrieval_accuracy(validation_questions_fp=validation_questions_fp, retrieval_fp=retrieval_fp,
                   comparison_fp='../../../data/retrieval_results/accuracy_retrieval_ipeds.xlsx')

# #%%
# Generate XWalk Accuracy
# validation_questions_fp = '../../../data/retrieval_results/xwalk_qa_questions.xlsx'
# retrieval_fp = '../../../data/retrieval_results/xwalk_retrieval.xlsx'
#
# retrieval_accuracy(validation_questions_fp=validation_questions_fp, retrieval_fp=retrieval_fp,
#                    comparison_fp='../../../data/retrieval_results/accuracy_retrieval_xwalk.xlsx')