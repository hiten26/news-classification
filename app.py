import sys
import pickle
from pathlib import Path

sys.path.append(str(Path.cwd()))

import streamlit as st
import sklearn

class_dict={
            1:'Company',
            2:'EducationalInstitution',
            3:'Artist',
            4:'Athlete',
            5:'OfficeHolder',
            6:'MeanOfTransportation',
            7:'Building',
            8:'NaturalPlace',
            9:'Village',
            10:'Animal',
            11:'Plant',
            12:'Album',
            13:'Film',
            14:'WrittenWork'
        }


@st.cache(allow_output_mutation=True)
def load_transform():
    tfv_ = pickle.load(open('/app/models/transform.pkl', 'rb'))
    return tfv_


@st.cache(allow_output_mutation=True)
def load_model():
    clf_ = pickle.load(open('/app/models/logistic_regression_model.pkl', 'rb'))
    return clf_


tfv = load_transform()
classifier = load_model()


def sample_predict(input_article):
    global class_dict, tfv, classifier
    article_tfv = tfv.transform([input_article])
    prediction = classifier.predict(article_tfv)[0]
    return class_dict[prediction]


st.title("News Categorization")


input_article_dd = st.selectbox("Example inputs", ('<select>', "Says Barack Obama is a socialist",
                                                   "In the days leading up to his trip to Cairo to give a long-awaited speech before a large Muslim audience, President Barack Obama gave several radio and television interviews, including one in which he claimed that the United States has so many Muslim residents that it would qualify as one of the largest populations in the world. If you actually took the number of Muslims Americans (sic), we'd be one of the largest Muslim countries in the world, he told French television station Canal Plus. Obama's comment was meant to emphasize the religious diversity of the United States and the importance of understanding between the Western world and followers of Islam, a key point in the speech he gave in Cairo on June 4, 2009.",
                                                    "Gov. Tom Wolf this week pushed for a severance tax on natural gas produced in Pennsylvania to help finance infrastructure improvements and a plan he hopes will put residents who lost their jobs in the pandemic back to work. He touted the proposed tax by describing Pennsylvania as an outlier in taxing natural gas drilling.  We’re the only major gas-producing state in the U.S. that doesn’t have a severance tax, he said, referring to a tax on raw materials extracted from the ground.",
                                                   "Over the last 10 years, incomes for the top 1 percent have grown Meanwhile, the bottom half of the country, they’ve seen their wages stagnate",
                                                   "Minnick voted to let the government fund abortion under Obamacare",
                                                   "The United States is the only developed country in the world that has not cut its corporate tax rate"))


if input_article_dd != '<select>':
    input_article = st.text_area("Enter article for prediction (English language only)", input_article_dd)
else:
    input_article = st.text_area("Enter article for prediction (English language only)")

if len(input_article) == 0 or input_article.isspace():
    st.warning('Please input a article.')
    st.stop()

if st.button("Submit"):
    prediction = sample_predict(input_article)
    st.write(prediction)

