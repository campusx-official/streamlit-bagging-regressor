import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

plt.style.use('seaborn-bright')

n_train = 150
n_test = 100
noise = 0.1

np.random.seed(0)
# Generate data
def f(x):
    x = x.ravel()
    return np.exp(-x ** 2) + 1.5 * np.exp(-(x - 2) ** 2)

def generate(n_samples, noise):
    X = np.random.rand(n_samples) * 10 - 5
    X = np.sort(X).ravel()
    y = np.exp(-X ** 2) + 1.5 * np.exp(-(X - 2) ** 2)\
        + np.random.normal(0.0, noise, n_samples)
    X = X.reshape((n_samples, 1))

    return X, y


X_train, y_train = generate(n_samples=n_train, noise=noise)
X_test, y_test = generate(n_samples=n_test, noise=noise)

st.sidebar.markdown("# Bagging Regressor")

estimator = st.sidebar.selectbox(
    'Select base estimator',
    ('Decision Tree', 'SVM', 'KNN')
)



n_estimators = int(st.sidebar.number_input('Enter number of estimators'))

max_samples = st.sidebar.slider('Max Samples', 0, 150, 150,step=25)

bootstrap_samples = st.sidebar.radio(
    "Bootstrap Samples",
    ('True', 'False')
)


# Load initial graph
fig, ax = plt.subplots()

# Plot initial graph
ax.scatter(X_train, y_train,color="yellow", edgecolor="black")
orig = st.pyplot(fig)

if st.sidebar.button('Run Algorithm'):

    if estimator == 'Decision Tree':
        algo = DecisionTreeRegressor()
        reg = DecisionTreeRegressor().fit(X_train, y_train)
    elif estimator == 'SVM':
        algo = SVR()
        reg = SVR().fit(X_train, y_train)
    else:
        algo = KNeighborsRegressor()
        reg = KNeighborsRegressor().fit(X_train, y_train)

    bag_reg = BaggingRegressor(algo,n_estimators=n_estimators,max_samples=max_samples,bootstrap=bootstrap_samples).fit(X_train, y_train)
    bag_reg_predict = bag_reg.predict(X_test)


    reg_predict = reg.predict(X_test)

    # R2 scores
    bag_r2 = r2_score(y_test,bag_reg_predict)
    reg_r2 = r2_score(y_test,reg_predict)

    orig.empty()

    fig, ax = plt.subplots()
    fig1, ax1 = plt.subplots()

    st.subheader("Bagging - " + estimator + " (R2 score - " + str(round(bag_r2,2)) + ")")
    ax1.scatter(X_train, y_train, color="yellow", edgecolor="black")
    ax1.plot(X_test, bag_reg_predict, linewidth=1, label="Bagging")
    ax1.legend()
    orig1 = st.pyplot(fig1)

    st.subheader(estimator  + " (R2 score - " + str(round(reg_r2,2)) + ")")
    ax.scatter(X_train, y_train, color="yellow", edgecolor="black")
    ax.plot(X_test, reg_predict, linewidth=1, color='red', label=estimator)
    ax.legend()
    orig = st.pyplot(fig)


