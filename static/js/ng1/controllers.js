'use strict';

controllers.controller('bodyController', ['$scope', '$log', '$http','$uibModal','fileUpload',
    function($scope, $log, $http, $uibModal, fileUpload) {

        var transform = function(data){
            return $.param(data);
        };

        console.log("bodyController::init");


        //-------- Analysis1
        $scope.analys1 = {'data':'analys1.data'};

        $scope.analys1func1 = function(){
          console.log($scope.analys1);

          var data = {
                param:JSON.stringify($scope.analys1)
            };

          $http({
                method  :'POST',
                url     :'/analys1',
                headers : {'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'},
                transformRequest: transform,
                data    : data
            })
                .then(function(res){
                            console.log("res.data",res.data);

                });
        };


        //loaded
        $scope.loaded ={
            csv_rows        :[],
            csv_trans_rows  :[],
            csv_column      :[],
            t_labels        :[]
        };
        
        //model
        $scope.model = {
            t:{
                selected    :0,
                colname     :'',
                modify      :0
            },
            x1:{
                selected    :1,
                colname     :''
            },
            x2:{
                selected    :2,
                colname     :''
            },
            eta             :0.25123,
            epoch           :50,
            random_state    :1,
            c               :10,
            penalty         :'l2',
            kernel          :'rbf',
            gamma           :2.45,
            test_size       :0.3,
            sample_scaling  :1,
            onehot_encode   :0,
            sample_max      :100,
            pre_analysis    :1,
            algorithm:[
                {type:0, name:'Linear Regression'},
                {type:1, name:'ADAlineSGD'},
                {type:2, name:'Support Vector Machine'},
                {type:3, name:'Logistic Regression'},
                {type:4, name:'Decision Tree'},
                {type:5, name:'K means Nearest Neighbors'}
            ]
        };
        
        //query
        $scope.query = {
            status      :0,
            file        :'',
            type        :'CSV',
            url         :'http://ocw.mit.edu/courses/sloan-school-of-management/15-097-prediction-machine-learning-and-statistics-spring-2012/datasets/wine.csv',
            header_is   :true,
            upload_is   :true,
            null_del_is :true,
            action      :'0',
            comment     :'Waiting.',
            algorithm_selected:$scope.model.algorithm[2]
        };

        $scope.sentiment = {
            rawtext     :"I really love this movie",
            label       :"",
            proba       :0.0,
            feedback    :-1,
            fontcolor   :'#23dea1'
        };

        $scope.sentiment.analyseAction = function(action){

            var param ={
                        action: action,
                        status: true,
                        review: $scope.sentiment.rawtext,
                        prediction: $scope.sentiment.label
                    };

            var data = {
                param:JSON.stringify(param)
            };

            $http({
                method  :'POST',
                url     :'/sentimentanalysis',
                headers : {'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'},
                transformRequest: transform,
                data    : data
            })
                .then(function(res){
                    console.log("type",res.data.action);
                    console.log("res.data",res.data);

                    if(res.data.action=='analyse') {
                        var proba = Math.round(res.data.proba * 100) +' %';
                        $scope.sentiment.proba = proba;
                        $scope.sentiment.label = res.data.label;
                        $scope.sentiment.fontcolor='#23dea1';
                    }
                    else if(res.data.action=='feedback') {
                        $scope.sentiment.feedback=-1;
                        $log.log("feedbaack");
                        var proba = Math.round(res.data.proba * 100) +' %';

                        console.log(proba);
                        $scope.sentiment.proba = proba;
                        $scope.sentiment.label = res.data.label;
                        $scope.sentiment.fontcolor='#de3423';
                    }
                    else{
                        $scope.sentiment.feedback=-1;
                         $log.log("not analyse");
                    }

                },
                function(res_error){
                    console.log(res_error);
                });

              //   .success(function(data, status, headers, config){
              //   $log.log(data);
              //   $log.log("success!");
              //   $scope.sentiment.feedback=-1;
              //

              // }).error(function(data, status, headers, config){
              //     $log.log("$http FAILED");
              // });
        };

        $scope.sentiment.feedbackAction = function(num){

            console.log('sentiment.feedbackAction = '+num);


        };

        $scope.step = {};
        $scope.step.dataLoadBtnAction = function(action){
            $scope.query.status = 1;
            var file=$scope.query.file;
            var uploadUrl = "/upload";
            var param={
                step            :'step1',
                action          :action,
                url             :$scope.query.url,
                header_is       :$scope.query.header_is,
                null_treatment  :$scope.query.null_del_is
            };
            fileUpload.uploadFileToUrl(file, uploadUrl,param,$scope);
        };

        $scope.step.sampleSelection = function(index) {
            $scope.query.status = 2;
            var tmp = $scope.loaded.csv_rows[Object.keys($scope.loaded.csv_rows)[0]];
            $scope.loaded.t_labels = $scope.loaded.csv_trans_rows[Object.keys($scope.loaded.csv_trans_rows)[$scope.model.t.selected]];
            $scope.model.x1.colname = Object.keys(tmp)[$scope.model.x1.selected];
            $scope.model.x2.colname = Object.keys(tmp)[$scope.model.x2.selected];
            $scope.model.t.colname  = Object.keys(tmp)[$scope.model.t.selected];
        };

        $scope.step.labelMapAction= function(index){
            $scope.query.status = 3;
            var tmp = $scope.loaded.csv_rows[Object.keys($scope.loaded.csv_rows)[0]];
            $scope.model.x1.colname = Object.keys(tmp)[$scope.model.x1.selected];
            $scope.model.x2.colname = Object.keys(tmp)[$scope.model.x2.selected];
        };

        $scope.step.testModelAction = function(){
            console.log("test_size:"+ $scope.model.test_size);
        };

        $scope.step.loadAlgorithm = function(){
            var param ={
                        step    :'step2',
                        status  :true,
                        message :'',
                        model   :$scope.model,
                        query   :$scope.query
                    };

            var data = {
                param:JSON.stringify(param)
            };

            $http({
                method  :'POST',
                url     :'/upload',
                headers : {'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'},
                transformRequest: transform,
                data    : data
            }).success(function(data, status, headers, config){
                $log.log(data);
                $log.log("success!!!");
                 $uibModal.open({
                        template: '<div class="md"><img src="/static/out/linear_regression1.png?rand='+Math.random()+'"/></div>'
                 })
              }).error(function(data, status, headers, config){
                  $log.log("$http FAILED");
            });
        };

        $scope.step.postLinearRegression = function(){
            var param={
                cgi_type    :'linear_regression',
                eta         :$scope.model.eta,
                epoch       :$scope.model.epoch
            };
            $http({
                method      :'POST',
                url         :'/cgi',
                headers     : {'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'},
                transformRequest: transform,
                data        : param
            }).success(function(data, status, headers, config){
                $log.log(data);
                 $uibModal.open({
                        template: '<div class="md"><img src="/static/out/linear_regression1.png?rand='+Math.random()+'"/></div>'
                 })
              }).error(function(data, status, headers, config){
                  $log.log("$http FAILED");
            });
        };
    }
]);

