#%%
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L12-v2")
vector_store = Chroma(embedding_function=embeddings,persist_directory='../data/cip_soc_db_storev2',collection_name='cip_soc_database')

#%%
questions = [
    "What is the average amount of federal grant aid awarded to full-time first-time undergraduates at Seton Hall University?",
    "How much do books and supplies cost for the 2023-24 academic year at North Central Michigan College?",
    "Does Miller-Motte College-Macon provide institutionally-controlled housing for students?",
    "What is the average net price for students awarded grant or scholarship aid at Lebanon Valley College for the 2021-22 academic year?",
    "How much is the typical housing charge for an academic year at Baker University?",
    "What are the published out-of-state tuition and fees for the University of Vermont in 2023-24?",
    "What is the average amount of student loans awarded to full-time first-time undergraduates at Albion College?",
    "How much are the typical food charges for the academic year at North Georgia Technical College?",
    "What percentage of students at The University of Texas at El Paso live on campus?",
    "How much do undergraduate application fees cost at Dordt University?",
    "What is the average net price for undergraduate students at Delaware Valley University with an income over $110,000?",
    "How much is the tuition payment plan at Georgetown University?",
    "What alternative tuition plans are offered by Rider University?",
    "What are the average amounts of federal, state, and local grant aid awarded at Missouri State University-Springfield?",
    "How much is the off-campus (not with family) room and board at Furman University for the 2023-24 academic year?",
    "What is the typical board charge for the academic year at the University of Houston-Downtown?",
    "Are there any graduate application fees at Trinity Bible College and Graduate School?",
    "What is the average Pell grant aid awarded to full-time first-time undergraduates at the University of Missouri-St Louis?",
    "What is the average amount of institutional grant aid awarded to full-time first-time undergraduates at Victor Valley College?",
    "How much are the typical room and board charges for the 2022-23 academic year at Massachusetts Bay Community College?",
    "What is the average amount of other federal grant aid awarded at Macomb Community College?",
    "How much is the off-campus (with family) other expenses for the 2023-24 academic year at Roberts Wesleyan University?",
    "What is the average net price for students awarded Title IV federal financial aid at Dordt University?",
    "What are the published in-district tuition and fees for the University of the Potomac-Washington DC Campus?",
    "How much do books and supplies cost for the 2022-23 academic year at Jackson College?",
    "What is the average amount of federal student loans awarded to undergraduate students at Tillamook Bay Community College?",
    "How much do off-campus (not with family) other expenses cost at Moberly Area Community College for 2023-24?",
    "Does Bushnell University offer any alternative tuition plans?",
    "What are the typical housing charges for the academic year at Edmonds College?",
    "How much is the undergraduate application fee at Richmond Community College?",
    "What is the average net price for undergraduate students awarded grant or scholarship aid at Brigham Young University-Idaho for the 2021-22 academic year?",
    "What is the average amount of federal, state, and local grant aid awarded to students at Southern University and A & M College?",
    "How much is the typical food charge for the academic year at the University of Vermont?",
    "What is the average amount of federal grant aid awarded to full-time first-time undergraduates at the University of West Alabama?",
    "What is the average amount of other student loans awarded at The University of Texas at El Paso?",
    "How much are the published in-state tuition and fees for Brown University in 2023-24?",
    "What is the average amount of Pell grant aid awarded to undergraduate students at Rider University?",
    "What are the typical room charges for the academic year at Jacksonville University?",
    "How much is the average net price for students awarded grant or scholarship aid at Indiana State University for the 2021-22 academic year?",
    "How much do books and supplies cost for the 2021-22 academic year at Lindsey Wilson College?",
    "What is the average net price for students with incomes over $110,000 at Rollins College?",
    "What is the published out-of-state tuition and fees for Baker University for the 2023-24 academic year?",
    "How much is the tuition payment plan at Otterbein University?",
    "What is the average amount of state/local grant aid awarded to full-time first-time undergraduates at Mississippi Gulf Coast Community College?",
    "What is the average net price for students awarded Title IV federal financial aid at Washington Adventist University for the 2020-21 academic year?",
    "How much do off-campus (not with family) room and board costs at Otterbein University for 2022-23?",
    "What is the average amount of institutional grant aid awarded to full-time first-time undergraduates at Polytechnic University of Puerto Rico-Miami?",
    "How much do typical housing charges for the academic year cost at the University of Wisconsin-Stout?",
    "What are the average amounts of federal, state, local, or institutional grant aid awarded to undergraduate students at Chapman University?",
    "How much do books and supplies cost for the 2020-21 academic year at Calvin University?",
    "What is the average net price for students awarded grant or scholarship aid at the University of Michigan-Dearborn?",
    "How much is the undergraduate application fee at the University of Houston-Downtown?",
    "What is the average amount of federal grant aid awarded to full-time first-time undergraduates at Husson University?",
    "What are the typical food charges for the academic year at San Diego State University?",
    "How much is the average amount of other federal grant aid awarded at Western Illinois University?",
    "What is the average net price for students with incomes between $0 and $30,000 at Everett Community College?",
    "What is the average amount of student loans awarded to full-time first-time undergraduates at the University of Science and Arts of Oklahoma?",
    "How much do off-campus (with family) other expenses cost at Jackson College for 2023-24?",
    "What is the average amount of Pell grant aid awarded to undergraduate students at Florida Atlantic University?",
    "How much do books and supplies cost for the 2021-22 academic year at Loyola University Chicago?",
    "What is the average net price for students with incomes between $48,001 and $75,000 at Turtle Mountain Community College?",
    "What is the published in-state tuition and fees for DeVry University-Texas in 2023-24?",
    "What is the average amount of other student loans awarded to full-time first-time undergraduates at Jacksonville University?",
    "How much do typical housing charges for the academic year cost at Buena Vista University?",
    "What are the average amounts of federal, state, and local grant aid awarded to students at Warner University?",
    "What is the average amount of institutional grant aid awarded to undergraduate students at Loyola University Chicago?",
    "How much are the typical food charges for the academic year at Averett University?",
    "What is the average net price for students with incomes over $110,000 at the University of St Thomas?",
    "How much is the graduate application fee at Reedley College?",
    "What is the average amount of federal student loans awarded to undergraduate students at Rollins College?",
    "What are the typical housing charges for the academic year at Randolph Community College?",
    "How much do off-campus (not with family) room and board costs at Concordia University-Wisconsin for 2022-23?",
    "What is the average amount of federal grant aid awarded to full-time first-time undergraduates at Community College of Beaver County?",
    "How much do books and supplies cost for the 2022-23 academic year at Baker University?",
    "What is the average net price for students awarded grant or scholarship aid at South College for the 2020-21 academic year?",
    "What is the average amount of state/local grant aid awarded to full-time first-time undergraduates at Alvin Community College?",
    "How much do off-campus (with family) other expenses cost at Massachusetts Bay Community College?",
    "What is the average amount of federal student loans awarded to undergraduate students at Dordt University?",
    "How much are the typical room charges for the academic year at Brigham Young University-Idaho?",
    "What is the average net price for students awarded Title IV federal financial aid at Modesto Junior College for 2021-22?",
    "What is the average amount of institutional grant aid awarded to students at South College?",
    "How much is the undergraduate application fee at Victor Valley College?",
    "What are the average amounts of federal, state, local, or institutional grant aid awarded to undergraduate students at the University of Arkansas?",
    "How much are the typical board charges for the academic year at Otterbein University?",
    "What is the average net price for students with incomes between $30,001 and $48,000 at Southern University at New Orleans?",
    "How much do books and supplies cost for the 2021-22 academic year at James A. Rhodes State College?",
    "What is the average amount of federal grant aid awarded to undergraduate students at East Central University?",
    "How much are the published out-of-state tuition and fees for the University of Houston-Downtown for 2023-24?",
    "What is the average net price for students with incomes over $110,000 at Southern Arkansas University Tech?",
    "How much do typical housing charges for the academic year cost at the University of St Thomas?",
    "What is the average amount of other federal grant aid awarded at Emerson College?",
    "How much is the graduate application fee at Husson University?",
    "What is the average amount of student loans awarded to full-time first-time undergraduates at Concordia University-Wisconsin?",
    "How much do off-campus (not with family) other expenses cost at Baker University for 2023-24?",
    "What is the average net price for students awarded grant or scholarship aid at the University of Vermont for the 2022-23 academic year?",
    "How much is the undergraduate application fee at Dordt University?",
    "What is the average amount of institutional grant aid awarded to full-time first-time undergraduates at the University of Michigan-Dearborn?",
    "How much do books and supplies cost for the 2020-21 academic year at Jackson College?",
    "What is the average amount of federal student loans awarded to undergraduate students at Washington Adventist University?",
    "How much are the typical room and board charges for the academic year at the University of Science and Arts of Oklahoma?"
]


#%%
returned_document = []
returned_metadata = []
for query in questions:
    answer = vector_store.similarity_search(query,k=1)
    returned_document.append(answer[0].page_content)
    returned_metadata.append(answer[0].metadata)

#%%

df = pd.DataFrame({'Query':questions,
                   'Returned_Document':returned_document,
                   'Returned_Metadata':returned_metadata})

df.head()

#%%
df.to_excel('../data/rag_retrieval_metrics.xlsx')


