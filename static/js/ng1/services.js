'use strict';

services.service('fileUpload', ['$http', function ($http) {
    this.uploadFileToUrl = function(file, uploadUrl,param,$scope){
       var fd = new FormData();
       fd.append('file', file);
       fd.append('param',JSON.stringify(param));
       $http.post(uploadUrl, fd, {
          //timeout: 3000,
          transformRequest: angular.identity,
          headers: {'Content-Type': undefined}
       })
       .success(function(data,status,headers,config){

           var status = data['status'] ? 'SUCCESS' : 'FAILED';
           $scope.query.status = data['status'] ? $scope.query.status : 0;
           $scope.query.comment = "[status] "+status+"\n";
           $scope.query.comment += "[message] "+data['message']+"\n";
           if(data['status']) {
               $scope.query.comment += "[shape] "+data['shape'][0]+" samples x "+data['shape'][1]+" features \n";
               $scope.query.comment += "[null_num] "+data['null_num']+"\n";
               if(data['null_num']>0){
                    var null_treatment = data['null_treatment'] ? 'Deleted' : 'Imputed with average';
                    $scope.query.comment += "[null_treatment] " + null_treatment + "\n";
               }
               var header_is = data['header_is'] ? '0th row' : 'no header';
               $scope.query.comment += "[header_is] " + header_is + "\n";
               $scope.loaded.csv_rows = JSON.parse(data['head']);
               $scope.loaded.csv_trans_rows = JSON.parse(data['head_t']);
               $scope.loaded.csv_column = JSON.parse(data['column']);
               $scope.query.url = data['url'];
           }
       })
       .error(function(data,status,headers,config){
           console.log('error');
       });
    }
}]);