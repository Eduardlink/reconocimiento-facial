-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Servidor: 127.0.0.1
-- Tiempo de generación: 08-01-2025 a las 05:27:50
-- Versión del servidor: 10.4.32-MariaDB
-- Versión de PHP: 8.2.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Base de datos: `reconocimiento_facial`
--

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `usuarios`
--

CREATE TABLE `usuarios` (
  `id` int(11) NOT NULL,
  `usuario` varchar(50) NOT NULL,
  `contraseña` varchar(255) NOT NULL,
  `url_carpeta` varchar(255) NOT NULL,
  `fecha_creacion` date NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `usuarios`
--

INSERT INTO `usuarios` (`id`, `usuario`, `contraseña`, `url_carpeta`, `fecha_creacion`) VALUES
(1, 'maria_gomez', 'password123', 'users/maria_gomez', '2025-01-02'),
(57, 'santa', '123', 'users/santa', '2025-01-07'),
(59, 'chris', '123', 'users/chris', '2025-01-07'),
(60, 'kikin', '123', 'users/kikin', '2025-01-07'),
(61, 'marlon', '123', 'users/marlon', '2025-01-07'),
(62, 'kate', '123', 'users/kate', '2025-01-07'),
(70, 'karen', '123', 'users/karen', '2025-01-07'),
(72, 'daniel', '123', 'users/daniel', '2025-01-07'),
(73, 'kevin', '123', 'users/kevin', '2025-01-07'),
(74, 'jose', '123', 'users/jose', '2025-01-07'),
(78, 'edu', '123', 'users/edu', '2025-01-07'),
(79, 'jaime', '123', 'users/jaime', '2025-01-07'),
(80, 'sofia', '123', 'users/sofia', '2025-01-07'),
(81, 'ana', '123', 'users/ana', '2025-01-07');

--
-- Índices para tablas volcadas
--

--
-- Indices de la tabla `usuarios`
--
ALTER TABLE `usuarios`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT de las tablas volcadas
--

--
-- AUTO_INCREMENT de la tabla `usuarios`
--
ALTER TABLE `usuarios`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=82;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
