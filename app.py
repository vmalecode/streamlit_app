import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load calculated columns
att_p = pd.read_csv('./data/attending_phy_count.csv', header=0, index_col=0, squeeze = True).to_dict()
att_op = pd.read_csv('./data/operating_phy_count.csv', header=0, index_col=0, squeeze = True).to_dict()
att_ot = pd.read_csv('./data/other_phy_count.csv', header=0, index_col=0, squeeze = True).to_dict()
prv_csv = pd.read_csv('./data/provider_phy_count.csv', header=0, index_col=0, squeeze = True).to_dict()
def convert(att_phy, opr_phy, other_phy):
    phy_total = 0
    if att_phy:
        count = att_p.get(att_phy)
        phy_total += count if count else 0
    if opr_phy:
        count = att_op.get(opr_phy)
        phy_total += count if count else 0
    if other_phy:
        count = att_ot.get(other_phy)
        phy_total += count if count else 0
    return phy_total

def convert2(provider):
    phy_total = 0
    if provider:
        count = prv_csv.get(provider)
        phy_total += count if count else 0
    return phy_total
@st.cache_data
def convert_df_to_csv(df):
  # IMPORTANT: Cache the conversion to prevent computation on every rerun
  return df.to_csv().encode('utf-8')

# Load the classifier model
clf = joblib.load('./model/iclassifier1.joblib')
imgUrl = "https://images.unsplash.com/photo-1495592822108-9e6261896da8?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1740&q=80"
# Define the Streamlit app
st.set_page_config(
     page_title="Insurance Fraud Detection",
)
def app():
    st.image(imgUrl, use_column_width="always")
    st.title('Insurance Fraud Detection')
    examples = dict()
    for i in range(1,9):
        examples['Example ' + str(i)] = 'example' + str(i) + '.csv'
    selected_example = st.selectbox('Download example CSV file:', list(examples.keys()))
    if selected_example:
        file_path = f'./test_data/{examples[selected_example]}'
        with open(file_path, 'rb') as f:
            file_contents = f.read()
        st.download_button(label='Download', data=file_contents, file_name=examples[selected_example])
    # Allow the user to upload a CSV file
    uploaded_file = st.file_uploader("Upload a CSV file with claim data", type=["csv"])
    used_fields = ['inscclaimamtreimbursed','deductibleamtpaid','gender','race','renaldiseaseindicator','state','county','noofmonths_partacov','noofmonths_partbcov','chroniccond_alzheimer','chroniccond_heartfailure','chroniccond_kidneydisease','chroniccond_cancer','chroniccond_obstrpulmonary','chroniccond_depression','chroniccond_diabetes','chroniccond_ischemicheart','chroniccond_osteoporasis','chroniccond_rheumatoidarthritis','chroniccond_stroke','ipannualreimbursementamt','ipannualdeductibleamt','opannualreimbursementamt','opannualdeductibleamt','inpt','ageatclaim','duration','los','dead','Att_Opr_Oth_Phy_Tot_Claims','Prv_Tot_Att_Opr_Oth_Phys']
    if uploaded_file is not None:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)
        df['Att_Opr_Oth_Phy_Tot_Claims'] = df.apply(lambda row: convert(row['attendingphysician'], row['operatingphysician'],row['otherphysician']), axis=1)
        df['Prv_Tot_Att_Opr_Oth_Phys'] = df.apply(lambda row: convert2(row['provider']), axis=1)
        # Get the data from the DataFrame as a list
        data = df.loc[:,used_fields]
        data.fillna(0,inplace=True)
        # Use the classifier to predict the entity type
        df['Investigate'] = clf.predict(data)
        # Display the models Investigate
        fraudulent = df.loc[df['Investigate'] == 1, ['claimid', 'Investigate']]
        distribution = df['Investigate'].value_counts()
        plt.rc('text', color='white')
        fig, ax = plt.subplots(facecolor='none')
        ax.pie(distribution, autopct='%1.1f%%')
        ax.legend(title='Legend', loc='best', labels=['Normal','Investigate'], framealpha = 0, bbox_to_anchor=(.90, 1))

        col1, col2 = st.columns(2, gap='medium')
        with col1:
            st.subheader("List of Potentially Fraudulent Claims")
            st.dataframe(fraudulent)
            st.download_button(
                label="Download data as CSV",
                data=convert_df_to_csv(fraudulent),
                file_name='fraudulent_claims.csv',
                mime='text/csv',
                )
        with col2:
            st.subheader("Ratio of Flagged Claims")
            st.text('')
            st.pyplot(fig, labels = ['False', 'True'])

# Run the Streamlit app
if __name__ == '__main__':
    app()
