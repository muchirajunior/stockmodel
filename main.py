from flask import Flask,render_template,request
from stockmodel import Model

app=Flask(__name__)

@app.route("/",methods=['GET','POST'])
def index():
    mydate=0
    myprice=0
    # Model.linescatterplot()
    # Model.bargraphplot()
    if request.method=="GET":
        if '' in request.args:
            x=0
            # name=request.args['name']

    if request.method=='POST':
        mydate=request.form['date']
        myprice=request.form['price']

    return render_template("index.html",normal_prediction=Model.normal_model(mydate,myprice),
    svr_lin_prediction=Model.svr_linear_model(mydate,myprice), svr_rbf_prediction=Model.svr_rbf_model(mydate,myprice),
    linear_prediction="not predicted")


if __name__=="__main__":
    app.run(debug=True,port=5000)
