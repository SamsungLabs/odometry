pipeline {
    agent {
        dockerfile {
            // additionalBuildArgs '--no-cache'
            args '--runtime=nvidia -v /datasets:/datasets -v /weights:/weights'
        }
    }
    stages {
        stage('Build') {
            steps {
                echo 'Building..'
                sh 'git status'
                sh 'cat .gitmodules'
                sh 'rm -rf weights'
                sh 'ln -s /weights weights'
            }
        }
        stage('Test') {
            steps {
                echo 'Testing..'
                sh '/home/odometry/miniconda/bin/activate /home/odometry/miniconda/envs/odometry'
                sh '/home/odometry/miniconda/envs/odometry/bin/python -m unittest discover -s tests -p "test_*.py"'
            }
        }
    }
}
