FROM nginx:1.27-alpine

# Remove config padrão
RUN rm /etc/nginx/conf.d/default.conf

# Copia config customizada
COPY nginx.conf /etc/nginx/nginx.conf

# Copia frontend estático
COPY static/ /etc/nginx/static/

EXPOSE 8082