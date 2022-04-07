import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def notify():
    msg = MIMEMultipart()
    msg['From'] = 'notifyto001@gmail.com'
    msg['To'] = 'ultimatrix1111@gmail.com'
    msg['Subject'] = 'Violence Detected'
    message = 'Please contact security, there is sever violence detected.'
    msg.attach(MIMEText(message))

    mailserver = smtplib.SMTP('smtp.gmail.com',587)
    # identify ourselves to smtp gmail client
    mailserver.ehlo()
    # secure our email with tls encryption
    mailserver.starttls()
    # re-identify ourselves as an encrypted connection
    mailserver.ehlo()
    mailserver.login('notifyto001@gmail.com', 'Notify1234')

    mailserver.sendmail('notifyto001@gmail.com','ultimatrix1111@gmail.com',msg.as_string())

    mailserver.quit()
