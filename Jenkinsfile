pipeline {
    agent {
        dockerfile {
        args '-v /datasets:/datasets'
        }
    }
    stages {
        stage('Build') {
            steps {
                echo 'Building..'
            }
        }
        stage('Test') {
            steps {
                echo 'Testing..'
                sh '~/miniconda/bin/activate ~/miniconda/envs/odometry'
                sh '~/miniconda/envs/odometry/bin/python -m unittest discover -s tests'
            }
        }
    }
}