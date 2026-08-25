FROM node:22-alpine AS build
WORKDIR /app
COPY frontend/package.json ./package.json
RUN npm install
COPY frontend ./
RUN npm run build

FROM nginx:1.27-alpine
COPY frontend/nginx.conf /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 8000
