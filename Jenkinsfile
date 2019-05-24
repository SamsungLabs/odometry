pipeline {
    agent {
        docker {image "odometry"}
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
                sh 'source activate ~/miniconda/envs/odometry'
                sh 'python -m unittest discover -s tests'
            }
        }
        stage('Deploy') {
            steps {
                echo 'Deploying....'
            }
        }
    }
}