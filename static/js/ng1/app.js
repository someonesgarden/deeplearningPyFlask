var controllers = angular.module('MainApp.ctrls', []);
var directives  = angular.module('MainApp.directives', []);
var services    = angular.module('MainApp.services', []);

angular.element(document).ready(
    function(){
        MainApp=angular.module('MainApp', ['ui.bootstrap', 'switcher', 'MainApp.directives', 'MainApp.ctrls', 'MainApp.services']);
        MainApp.config(['$httpProvider',
            function($httpProvider) {
                $httpProvider.defaults.timeout = 2000;
            }
        ]);
        angular.bootstrap(document, ['MainApp']);
        console.log("app.js::ready");
    }
);