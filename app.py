import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load calculated columns
att_p = pd.read_csv('data/attending_phy_count.csv', header=0, index_col=0, squeeze = True).to_dict()
att_op = pd.read_csv('data/operating_phy_count.csv', header=0, index_col=0, squeeze = True).to_dict()
att_ot = pd.read_csv('data/other_phy_count.csv', header=0, index_col=0, squeeze = True).to_dict()
prv_csv = pd.read_csv('data/provider_phy_count.csv', header=0, index_col=0, squeeze = True).to_dict()
imgUrl = "https://images.unsplash.com/photo-1495592822108-9e6261896da8?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1740&q=80"

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
clf = joblib.load('./model/gbrt991.pkl')

def app():
    # Allow the user to upload a CSV file
    def csv():
        examples = dict()
        for i in range(1,9):
            examples['Example ' + str(i)] = 'test_data/example' + str(i) + '.csv'
        selected_example = st.selectbox('Download example CSV file:', list(examples.keys()))
        if selected_example:
            file_path = f'{examples[selected_example]}'
            with open(file_path, 'rb') as f:
                file_contents = f.read()
            st.download_button(label='Download', data=file_contents, file_name=examples[selected_example])
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
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
            df['Potential_Fraud'] = clf.predict(data)
        # Display the models Potential_Fraud
            fraudulent = df.loc[df['Potential_Fraud'] == 1, ['claimid', 'Potential_Fraud']]
            distribution = df['Potential_Fraud'].value_counts()
            plt.rc('text', color='white')
            fig, ax = plt.subplots(facecolor='none')
            ax.pie(distribution, autopct='%1.1f%%')
            ax.legend(title='Legend', loc='best', labels=['Not Fraud','Potential Fraud'], framealpha = 0, bbox_to_anchor=(.90, 1))

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
                st.pyplot(fig)
    

    def forms():
        with st.form("medi_claim_form"):
            st.write("Please fill out the medi claim form below:")
            att_phy = st.text_input("Attending Physician")
            opr_phy = st.text_input("Operating Physician")
            other_phy = st.text_input("Other Physician")
            provider = st.text_input("Provider")
            gender = st.selectbox("gender", [0,1])
            race = st.selectbox("race", [1,2,3,5])
            state = st.selectbox("state", list(range(1,51)))
            county = st.selectbox("county", list(range(1,51)))
            ageatclaim = st.number_input("Age at claim")
            dead  = 1 if st.checkbox("dead") else 0
            duration = st.number_input("Duration")
            noofmonths_partacov = st.selectbox("Part A Coverage (in months)", list(range(1,13)))
            noofmonths_partbcov = st.selectbox("Part B Coverage (in months)", list(range(1,13)))
            renaldiseaseindicator = 1 if st.checkbox("renaldiseaseindicator") else 0
            chroniccond_alzheimer = 1 if st.checkbox("chroniccond_alzheimer") else 0
            chroniccond_heartfailure  = 1 if st.checkbox("chroniccond_heartfailure") else 0
            chroniccond_kidneydisease  = 1 if st.checkbox("chroniccond_kidneydisease") else 0
            chroniccond_cancer  = 1 if st.checkbox("chroniccond_cancer") else 0
            chroniccond_obstrpulmonary  = 1 if st.checkbox("chroniccond_obstrpulmonary") else 0
            chroniccond_depression  = 1 if st.checkbox("chroniccond_depression") else 0
            chroniccond_diabetes  = 1 if st.checkbox("chroniccond_diabetes") else 0
            chroniccond_ischemicheart  = 1 if st.checkbox("chroniccond_ischemicheart") else 0
            chroniccond_osteoporasis  = 1 if st.checkbox("chroniccond_osteoporasis") else 0
            chroniccond_rheumatoidarthritis  = 1 if st.checkbox("chroniccond_rheumatoidarthritis") else 0
            chroniccond_stroke  = 1 if st.checkbox("chroniccond_stroke") else 0
            inscclaimamtreimbursed = st.slider("Insurance Claim Amount reimbursed", min_value=0, max_value=10000, value=0, step=1)
            deductibleamtpaid = st.slider("Deductible Amount Paid", min_value=0, max_value=10000, value=0, step=1)
            
            inpt  = 1 if st.checkbox("In Patient") else 0
            los = st.number_input("Length of stay")
            ipannualreimbursementamt = st.slider("In patient Annual Reimbursement Amount", min_value=0, max_value=10000, value=0, step=1)
            ipannualdeductibleamt = st.slider("In patient annual Deductible Amount", min_value=0, max_value=10000, value=0, step=1)
            opannualreimbursementamt = st.slider("Out patient annual Reimbursement Amount", min_value=0, max_value=10000, value=0, step=1)
            opannualdeductibleamt = st.slider("Out patient annual Deductible Amount", min_value=0, max_value=10000, value=0, step=1)

            submit_button = st.form_submit_button("Submit")

            if submit_button:
                phy_tot = convert(att_phy,opr_phy,other_phy)
                prv_tot = convert2(provider)
                data = pd.DataFrame({
                    'inscclaimamtreimbursed' : [inscclaimamtreimbursed],
                    'deductibleamtpaid': [deductibleamtpaid],
                    'gender': [gender],
                    'race': [race],
                    'renaldiseaseindicator': [renaldiseaseindicator],
                    'state':[state],
                    'county': [county],
                    'noofmonths_partacov' : [noofmonths_partacov],
                    'noofmonths_partbcov' : [noofmonths_partbcov],
                    'chroniccond_alzheimer': [chroniccond_alzheimer],
                    'chroniccond_heartfailure': [chroniccond_heartfailure],
                    'chroniccond_kidneydisease' : [chroniccond_kidneydisease],
                    'chroniccond_cancer' : [chroniccond_cancer],
                    'chroniccond_obstrpulmonary' : [chroniccond_obstrpulmonary],
                    'chroniccond_depression': [chroniccond_depression],
                    'chroniccond_diabetes' : [chroniccond_diabetes],
                    'chroniccond_ischemicheart' : [chroniccond_ischemicheart],
                    'chroniccond_osteoporasis' : [chroniccond_osteoporasis],
                    'chroniccond_rheumatoidarthritis' : [chroniccond_rheumatoidarthritis],
                    'chroniccond_stroke' : [chroniccond_stroke],
                    'ipannualreimbursementamt' : [ipannualreimbursementamt],
                    'ipannualdeductibleamt' : [ipannualdeductibleamt],
                    'opannualreimbursementamt' : [opannualreimbursementamt],
                    'opannualdeductibleamt' : [opannualdeductibleamt],
                    'inpt' : [inpt],
                    'ageatclaim' : [ageatclaim],
                    'duration' : [duration],
                    'los' : [los],
                    'dead' : [dead],
                    'Att_Opr_Oth_Phy_Tot_Claims' : [phy_tot],
                    'Prv_Tot_Att_Opr_Oth_Phys' : [prv_tot]
                })
                data.fillna(0,inplace=True)
                if clf.predict(data)[0] == 0:
                    st.text("Not Fraud")
                elif clf.predict(data)[0] == 1:
                    st.text("Potential Fraud")
    st.image(imgUrl, use_column_width="always")
    st.title('Insurance Fraud Detection')
    page = st.sidebar.selectbox('Select page',['CSV','Form']) 

    if page == 'Form':
        forms()
    if page == 'CSV':
        csv()

# Run the Streamlit app
if __name__ == '__main__':
    app()