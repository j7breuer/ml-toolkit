pipeline {
    agent any

    environment {
        VERSION = "",
        LIBRARY_NAME = "ml-toolkit",
        PYPI_ENV = "local",
        LOCAL_PYPI = ""
    }

    stages {
        stage('Environment Setup') {
            steps {
                withPythonEnv('python3.9') {
                    echo '\n=======================\n[START] Initializing...\n=======================\n'
                    echo "Running ${env.BUILD_ID} on ${env.JENKINS_URL} \n"
                    echo "\n<--------- Installing PyTorch... --------->"
                    sh 'pip3.9 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu'
                    echo "\n<--------- Installing requirements.txt --------->"
                    sh "pip3.9 install -r requirements.txt --no-cache-dir --index-url http://\$NEXUS_USERNAME:\$NEXUS_PASSWORD@192.168.50.25:8081/repository/Workstation_PyPi/simple --trusted-host ${env.NEXUS}"
                    echo '\n=====================\n[END] Initializing...\n=====================\n'                    
                }
            }
        }
        stage('PyTest Unit Tests') {
            steps {
                withPythonEnv('python3.9'){
                    echo '\n============================\n[START] PyTest Unit Tests...\n============================\n'
                    echo '\n<--------- Running pytest... --------->'
                    sh 'python3.9 -m pytest --cov . --cov-report xml'
                    echo '\n==========================\n[END] PyTest Unit Tests...\n==========================\n'
                }
            }
        }
        stage('Sonar Scans') {
            environment {
                scannerHome = tool 'SonarQubeScanner-4.7.0'
            }
            steps {
                withSonarQubeEnv('SonarQube-8.3.1') {
                    echo '\n============================\n[START] Sonar Scans...\n============================\n'
                    sh '/var/lib/jenkins/tools/hudson.plugins.sonar.SonarRunnerInstallation/SonarQubeScanner-4.7.0/bin/sonar-scanner'
                    echo '\n============================\n[END] Sonar Scans...\n============================\n'
                }
            }
        }
        stage('Build Package') {
            steps {
                withPythonEnv('python3.9') {
                    script {
                        // Set version suffix based on branch name
                        def versionSuffix = ''

                        // Check if beta branch
                        if (env.BRANCH_NAME == 'beta') {
                            versionSuffix = '-beta'
                        }

                        // Pass version suffix as an environment variable
                        withEnv(["VERSION_SUFFIX=${versionSuffix}"]) {
                            echo '\n============================\n[START] Building Package...\n============================\n'
                            echo "\n<--------- Building package with version suffix: ${versionSuffix} --------->"
                            sh 'python3.9 setup.py sdist bdist_wheel'
                            echo '\n============================\n[END] Building Package...\n============================\n'
                        }
                    }
                }
            }
        }
        stage('Push to PyPi') {
            when {
                expression { return env.BRANCH_NAME == 'master' || env.BRANCH_NAME == 'beta'}
            }
            steps {
                script {
                    if (env.PYPI_ENV == 'local') {
                        echo "Pushing to local Nexus PyPI"
                        sh """
                        python -m twine upload --repository-url $LOCAL_PYPI \
                        --username $NEXUS_USERNAME --password $NEXUS_PASSWORD \
                        dist/*
                        """
                    } else if (env.PYPI_ENV == 'public') {
                        echo "Pushing to public PyPI"
                        sh """
                        python -m twine upload \
                        --username $PUBLIC_PYPI_USERNAME --password $PUBLIC_PYPI_PASSWORD \
                        dist/*
                        """
                    } else {
                        error "Invalid PYPI_ENV value. Must be 'local' or 'public'."
                    }
                }
            }
        }
    }
}