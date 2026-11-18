Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 6 items.

=== ITEM id=800ee6796db9e648 ===
Gold sentences and facts:
Sentence 0: The trade winds are the prevailing pattern of easterly surface winds found in the tropics , within the lower portion of the Earth 's atmosphere , in the lower section of the troposphere near the Earth 's equator . The trade winds blow predominantly from the northeast in the Northern Hemisphere and from the southeast in the Southern Hemisphere , strengthening during the winter and when the Arctic oscillation is in its warm phase . Trade winds have been used by captains of sailing ships to cross the world 's oceans for centuries , and enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans .
  Fact 0.0: Trade winds are the prevailing pattern of easterly surface winds in the tropics.
  Fact 0.1: Trade winds are found within the lower portion of the Earth's atmosphere, in the lower section of the troposphere.
  Fact 0.2: Trade winds are found near the Earth's equator.
  Fact 0.3: Trade winds blow predominantly from the northeast in the Northern Hemisphere.
  Fact 0.4: Trade winds blow predominantly from the southeast in the Southern Hemisphere.
  Fact 0.5: Trade winds strengthen during the winter.
  Fact 0.6: Trade winds strengthen when the Arctic oscillation is in its warm phase.
  Fact 0.7: Captains of sailing ships have used trade winds to cross the world's oceans for centuries.
  Fact 0.8: Trade winds enabled European empire expansion into the Americas.
  Fact 0.9: Trade winds enabled trade routes to become established across the Atlantic Ocean.
  Fact 0.10: Trade winds enabled trade routes to become established across the Pacific Ocean.

Retrieved items:
<item 1>
Trade winds
The subsident ( sinking ) air is relatively dry because as it descends , the temperature increases , but the absolute humidity remains constant , which lowers
the relative humidity of the air mass . This warm , dry air is known as a superior air mass and normally resides above a maritime tropical ( warm and moist )
air mass . An increase of temperature with height is known as a temperature inversion . When it occurs within a trade wind regime , it is known as a trade wind
inversion . The surface air that flows from these subtropical high - pressure belts toward the Equator is deflected toward the west in both hemispheres by the
Coriolis effect . These winds blow predominantly from the northeast in the Northern Hemisphere and from the southeast in the Southern Hemisphere . Because winds
are named for the direction from which the wind is blowing , these winds are called the northeasterly trade winds in the Northern Hemisphere and the
southeasterly trade winds in the Southern Hemisphere . The trade winds of both hemispheres meet at the doldrums . As they blow across tropical regions , air
masses heat up over lower latitudes due to more direct sunlight .
</item 1>

<item 2>
Trade winds
Encyclopedia of world climatology . Springer . p. 128 . ISBN 978 - 1 - 4020 - 3264 - 6 . Retrieved 2009 - 11 - 09 . Jump up ^ Bob Rauber ( 2009 - 05 - 22 ) . ``
Research - The Rain in Cumulus over the Ocean Campaign '' . Retrieved 2009 - 11 - 08 . Jump up ^ James P. Terry ( 2007 ) . Tropical cyclones : climatology and
impacts in the South Pacific . Springer . p. 8 . ISBN 978 - 0 - 387 - 71542 - 1 . Retrieved 2009 - 11 - 08 . Jump up ^ G.E. Pieter & F. Augustinus . `` The
influence of the trade winds on the coastal development of the Guianas at various scale levels : a synthesis '' . Marine Geology. 208 ( 2 - 4 ) : 145 -- 151 .
Bibcode : 2004MGeol. 208 ... 145A .
</item 2>

<item 3>
Trade winds
`` trade winds '' . Glossary of Meteorology . American Meteorological Society . Archived from the original on 2008 - 12 - 11 . Retrieved 2008 - 09 - 08 . Jump
up ^ Ralph Stockman Tarr and Frank Morton McMurry ( 1909 ). Advanced geography . W.W. Shannon , State Printing , pp. 246 . Retrieved on 2009 - 04 - 15 . Jump up
^ JetStream ( 2008 ) . `` How to read weather maps '' . National Weather Service . Retrieved 2007 - 05 - 16 . Jump up ^ Glossary of Meteorology ( June 2000 ) .
`` Tropical air '' . American Meteorological Society . Retrieved 2009 - 10 - 28 . Jump up ^ Glossary of Meteorology ( June 2000 ) . `` Trade air '' . American
Meteorological Society . Retrieved 2009 - 10 - 28 . Jump up ^ John E. Oliver ( 2005 ) . Encyclopedia of world climatology . Springer .
</item 3>

<item 4>
Trade winds
ISBN 978 - 1 - 4437 - 2300 - 8 . Retrieved 2009 - 11 - 09 . Jump up ^ Glossary of Meteorology ( June 2000 ) . `` Doldrums '' . American Meteorological Society .
Retrieved 2009 - 11 - 09 . Jump up ^ Glossary of Meteorology ( June 2000 ) . `` Intertropical Convergence Zone '' . American Meteorological Society . Retrieved
2009 - 11 - 09 . Jump up ^ Glossary of Meteorology ( June 2000 ) . `` Monsoon Trough '' . American Meteorological Society . Archived from the original on 2009 -
06 - 17 . Retrieved 2009 - 11 - 09 . Jump up ^ Glossary of Meteorology ( June 2000 ) . `` Superior air '' . American Meteorological Society . Retrieved 2009 -
10 - 28 . Jump up ^ Glossary of Meteorology ( 2009 ) . `` trade winds '' . Glossary of Meteorology .
</item 4>

<item 5>
Trade winds
Those that develop over land ( continental ) are drier and hotter than those that develop over oceans ( maritime ) , and travel northward on the western
periphery of the subtropical ridge . Maritime tropical air masses are sometimes referred to as trade air masses . The one region of the Earth which has an
absence of trade winds is the north Indian ocean . Weather effects ( edit ) Clouds which form above regions within trade wind regimes are typically composed of
cumulus which extend no more than 4 kilometres ( 13,000 ft ) in height , and are capped from being taller by the trade wind inversion . Trade winds originate
more from the direction of the poles ( northeast in the Northern Hemisphere , southeast in the Southern Hemisphere ) during the cold season , and are stronger
in the winter than the summer . As an example , the windy season in the Guianas , which lie at low latitudes in South America , occurs between January and April
. When the phase of the Arctic oscillation ( AO ) is warm , trade winds are stronger within the tropics . The cold phase of the AO leads to weaker trade winds .
</item 5>

<item 6>
Trade winds
Trade winds - wikipedia Trade winds Jump to : navigation , search This article is about the weather phenomenon . For other uses , see Tradewind . The westerlies
( blue arrows ) and trade winds ( yellow and brown arrows ) The trade winds are the prevailing pattern of easterly surface winds found in the tropics , within
the lower portion of the Earth 's atmosphere , in the lower section of the troposphere near the Earth 's equator . The trade winds blow predominantly from the
northeast in the Northern Hemisphere and from the southeast in the Southern Hemisphere , strengthening during the winter and when the Arctic oscillation is in
its warm phase . Trade winds have been used by captains of sailing ships to cross the world 's oceans for centuries , and enabled European empire expansion into
the Americas and trade routes to become established across the Atlantic and Pacific oceans . In meteorology , the trade winds act as the steering flow for
tropical storms that form over the Atlantic , Pacific , and southern Indian Oceans and make landfall in North America , Southeast Asia , and Madagascar and
eastern Africa , respectively . Trade winds also transport African dust westward across the Atlantic Ocean into the Caribbean Sea , as well as portions of
southeastern North America .
</item 6>

<item 7>
Trade winds
Between 1847 and 1849 , Matthew Fontaine Maury collected enough information to create wind and current charts for the world 's oceans . Cause ( edit ) 3D map
showing Hadley cells in relationship to trade winds on the surface . See also : Air mass , Hadley cell , Humidity , Intertropical Convergence Zone , Monsoon ,
Monsoon trough , Near - equatorial trough , and Prevailing winds As part of the Hadley cell circulation , surface air flows toward the equator while the flow
aloft is towards the poles . A low - pressure area of calm , light variable winds near the equator is known as the doldrums , near - equatorial trough ,
intertropical front , or the Intertropical Convergence Zone . When located within a monsoon region , this zone of low pressure and wind convergence is also
known as the monsoon trough . Around 30 ° in both hemispheres , air begins to descend toward the surface in subtropical high - pressure belts known as
subtropical ridges .
</item 7>

<item 8>
Trade winds
See also ( edit ) Intertropical Convergence Zone Winds in the Age of Sail Westerly wind burst Volta do mar References ( edit ) Jump up ^ Carol G. Braham ; Enid
Pearsons ; Deborah M. Posner ; Georgia S. Maas & Richard Goodman ( 2001 ) . Random House Webster 's College Dictionary ( second ed . ) . Random House . p. 1385
. ISBN 0 - 375 - 42560 - 8 . Jump up ^ Hermann R. Muelder ( 2007 ) . Years of This Land - A Geographical History of the United States . Read Books . p. 38 .
ISBN 978 - 1 - 4067 - 7740 - 6 . Retrieved 2009 - 11 - 09 . ^ Jump up to : Derek Hayes ( 2001 ) . Historical atlas of the North Pacific Ocean : maps of
discovery and scientific exploration , 1500 - 2000 . Douglas & McIntyre . p. 18 . ISBN 978 - 1 - 55054 - 865 - 5 .
</item 8>

<item 9>
Trade winds
They also learned that to reach South Africa , they needed to go far out in the ocean , head for Brazil , and around 30 ° S go east again . Following the
African coast southbound means upwind in the Southern hemisphere . In the Pacific ocean , the full wind circulation , which included both the trade wind
easterlies and higher - latitude Westerlies , was unknown to Europeans until Andres de Urdaneta 's voyage in 1565 . The captain of a sailing ship seeks a course
along which the winds can be expected to blow in the direction of travel . During the Age of Sail , the pattern of prevailing winds made various points of the
globe easy or difficult to access , and therefore had a direct effect on European empire - building and thus on modern political geography . For example ,
Manila galleons could not sail into the wind at all . By the 18th century the importance of the trade winds to England 's merchant fleet for crossing the
Atlantic Ocean had led both the general public and etymologists to identify the name with a later meaning of ' trade ' , `` ( foreign ) commerce '' .
</item 9>

<item 10>
Trade winds
Shallow cumulus clouds are seen within trade wind regimes , and are capped from becoming taller by a trade wind inversion , which is caused by descending air
aloft from within the subtropical ridge . The weaker the trade winds become , the more rainfall can be expected in the neighboring landmasses . Contents ( hide
) 1 History 2 Cause 3 Weather effects 4 See also 5 References History ( edit ) A Spanish galleon See also : Age of Discovery , Volta do Mar , and Age of sail
The term trade winds originally derives from the early fourteenth century late Middle English word ' trade , ' meaning `` path '' or `` track . '' The
Portuguese recognized the importance of the trade winds ( then the Volta do mar , meaning in Portuguese `` turn of the sea '' but also `` return from the sea ''
) in navigation in both the north and south Atlantic ocean as early as the 15th century . From West Africa , the Portuguese had to sail away from continental
Portugal , that is , to west and northwest . They could then turn northeast , to the area around the Azores islands , and finally east to mainland Europe .
</item 10>
=== END ITEM id=800ee6796db9e648 ===

=== ITEM id=832a1d27bc6eeb3a ===
Gold sentences and facts:
Sentence 0: Year Laureate Country Rationale 1901 Wilhelm Conrad Röntgen Germany `` in recognition of the extraordinary services he has rendered by the discovery of the remarkable rays subsequently named after him '' 1902 Hendrik Lorentz Netherlands `` in recognition of the extraordinary service they rendered by their researches into the influence of magnetism upon radiation phenomena '' Pieter Zeeman Netherlands 1903 Antoine Henri Becquerel France `` for his discovery of spontaneous radioactivity '' Pierre Curie France `` for their joint researches on the radiation phenomena discovered by Professor Henri Becquerel '' Maria Skłodowska - Curie Poland France 1904 Lord Rayleigh United Kingdom `` for his investigations of the densities of the most important gases and for his discovery of argon in connection with these studies '' 1905 Philipp Eduard Anton von Lenard Austria - Hungary Germany `` for his work on cathode rays '' 1906 Joseph John Thomson United Kingdom `` for his theoretical and experimental investigations on the conduction of electricity by gases '' 1907 Albert Abraham Michelson United States Poland `` for his optical precision instruments and the spectroscopic and metrological investigations carried out with their aid '' 1908 Gabriel Lippmann France `` for his method of reproducing colours photographically based on the phenomenon of interference '' 1909 Guglielmo Marconi Italy `` for their contributions to the development of wireless telegraphy '' Karl Ferdinand Braun Germany 1910 Johannes Diderik van der Waals Netherlands `` for his work on the equation of state for gases and liquids '' 1911 Wilhelm Wien Germany `` for his discoveries regarding the laws governing the radiation of heat '' 1912 Nils Gustaf Dalén Sweden `` for his invention of automatic valves designed to be used in combination with gas accumulators in lighthouses and buoys '' 1913 Heike Kamerlingh - Onnes Netherlands `` for his investigations on the properties of matter at low temperatures which led , inter alia , to the production of liquid helium '' 1914 Max von Laue Germany `` For his discovery of the diffraction of X-rays by crystals '' , an important step in the development of X-ray spectroscopy . 1915 William Henry Bragg United Kingdom `` For their services in the analysis of crystal structure by means of X-rays '' , an important step in the development of X-ray crystallography William Lawrence Bragg Australia United Kingdom 1916 Not awarded World War I 1917 Charles Glover Barkla United Kingdom `` For his discovery of the characteristic Röntgen radiation of the elements '' , another important step in the development of X-ray spectroscopy 1918 Max Planck Germany `` for the services he rendered to the advancement of physics by his discovery of energy quanta '' 1919 Johannes Stark Germany `` for his discovery of the Doppler effect in canal rays and the splitting of spectral lines in electric fields '' 1920 Charles Édouard Guillaume Switzerland `` for the service he has rendered to precision measurements in physics by his discovery of anomalies in nickel - steel alloys '' 1921 Albert Einstein Germany Switzerland `` for his services to theoretical physics , and especially for his discovery of the law of the photoelectric effect '' 1922 Niels Bohr Denmark `` for his services in the investigation of the structure of atoms and of the radiation emanating from them '' 1923 Robert Andrews Millikan United States `` for his work on the elementary charge of electricity and on the photoelectric effect '' 1924 Manne Siegbahn Sweden `` for his discoveries and research in the field of X-ray spectroscopy '' 1925 James Franck Germany `` for their discovery of the laws governing the impact of an electron upon an atom '' Gustav Hertz Germany 1926 Jean Baptiste Perrin France `` for his work on the discontinuous structure of matter , and especially for his discovery of sedimentation equilibrium '' 1927 Arthur Holly Compton United States `` for his discovery of the effect named after him '' Charles Thomson Rees Wilson United Kingdom `` for his method of making the paths of electrically charged particles visible by condensation of vapour '' 1928 Owen Willans Richardson United Kingdom `` for his work on the thermionic phenomenon and especially for the discovery of the law named after him '' 1929 Louis Victor Pierre Raymond , 7th Duc de Broglie France `` for his discovery of the wave nature of electrons '' 1930 Chandrasekhara Venkata Raman India `` for his work on the scattering of light and for the discovery of the effect named after him '' 1931 Not awarded 1932 Werner Heisenberg Germany `` for the creation of quantum mechanics , the application of which has , inter alia , led to the discovery of the allotropic forms of hydrogen '' 1933 Erwin Schrödinger Austria `` for the discovery of new productive forms of atomic theory '' Paul Dirac United Kingdom 1934 Not awarded 1935 James Chadwick United Kingdom `` for the discovery of the neutron '' 1936 Victor Francis Hess Austria `` for his discovery of cosmic radiation '' Carl David Anderson United States `` for his discovery of the positron '' 1937 Clinton Joseph Davisson United States `` for their experimental discovery of the diffraction of electrons by crystals '' George Paget Thomson United Kingdom 1938 Enrico Fermi Italy `` for his demonstrations of the existence of new radioactive elements produced by neutron irradiation , and for his related discovery of nuclear reactions brought about by slow neutrons '' 1939 Ernest Lawrence United States `` for the invention and development of the cyclotron and for results obtained with it , especially with regard to artificial radioactive elements '' 1940 Not awarded World War II 1941 Not awarded World War II 1942 Not awarded World War II 1943 Otto Stern United States `` for his contribution to the development of the molecular ray method and his discovery of the magnetic moment of the proton '' 1944 Isidor Isaac Rabi United States Poland `` for his resonance method for recording the magnetic properties of atomic nuclei '' 1945 Wolfgang Pauli Austria `` for the discovery of the Exclusion Principle , also called the Pauli principle '' 1946 Percy Williams Bridgman United States `` for the invention of an apparatus to produce extremely high pressures , and for the discoveries he made there within the field of high pressure physics '' 1947 Edward Victor Appleton United Kingdom `` for his investigations of the physics of the upper atmosphere especially for the discovery of the so - called Appleton layer '' 1948 Patrick Maynard Stuart Blackett United Kingdom `` for his development of the Wilson cloud chamber method , and his discoveries therewith in the fields of nuclear physics and cosmic radiation '' 1949 Hideki Yukawa Japan `` for his prediction of the existence of mesons on the basis of theoretical work on nuclear forces '' 1950 Cecil Frank Powell United Kingdom `` for his development of the photographic method of studying nuclear processes and his discoveries regarding mesons made with this method '' 1951 John Douglas Cockcroft United Kingdom `` for their pioneer work on the transmutation of atomic nuclei by artificially accelerated atomic particles '' Ernest Thomas Sinton Walton Ireland 1952 Felix Bloch Switzerland United States `` for their development of new methods for nuclear magnetic precision measurements and discoveries in connection therewith '' Edward Mills Purcell United States 1953 Frits Zernike Netherlands `` for his demonstration of the phase contrast method , especially for his invention of the phase contrast microscope '' 1954 Max Born West Germany `` for his fundamental research in quantum mechanics , especially for his statistical interpretation of the wavefunction '' Walther Bothe West Germany `` for the coincidence method and his discoveries made therewith '' 1955 Willis Eugene Lamb United States `` for his discoveries concerning the fine structure of the hydrogen spectrum '' Polykarp Kusch United States Germany `` for his precision determination of the magnetic moment of the electron '' 1956 John Bardeen United States `` for their researches on semiconductors and their discovery of the transistor effect '' Walter Houser Brattain United States William Bradford Shockley United States 1957 Tsung - Dao Lee Republic of China `` for their penetrating investigation of the so - called parity laws which has led to important discoveries regarding the elementary particles '' Chen Ning Yang Republic of China 1958 Pavel Alekseyevich Cherenkov Soviet Union `` for the discovery and the interpretation of the Cherenkov effect '' Ilya Frank Soviet Union Igor Yevgenyevich Tamm Soviet Union 1959 Emilio Gino Segrè Italy United States `` for their discovery of the antiproton '' Owen Chamberlain United States 1960 Donald Arthur Glaser United States `` for the invention of the bubble chamber '' 1961 Robert Hofstadter United States `` for his pioneering studies of electron scattering in atomic nuclei and for his thereby achieved discoveries concerning the structure of the nucleons '' Rudolf Ludwig Mössbauer West Germany `` for his researches concerning the resonance absorption of gamma radiation and his discovery in this connection of the effect which bears his name '' 1962 Lev Davidovich Landau Soviet Union `` for his pioneering theories for condensed matter , especially liquid helium '' 1963 Eugene Paul Wigner Hungary United States `` for his contributions to the theory of the atomic nucleus and the elementary particles , particularly through the discovery and application of fundamental symmetry principles '' Maria Goeppert - Mayer United States `` for their discoveries concerning nuclear shell structure '' J. Hans D. Jensen West Germany 1964 Nicolay Gennadiyevich Basov Soviet Union `` for fundamental work in the field of quantum electronics , which has led to the construction of oscillators and amplifiers based on the maser -- laser principle '' Alexander Prokhorov Soviet Union Charles Hard Townes United States 1965 Richard Phillips Feynman United States `` for their fundamental work in quantum electrodynamics ( QED ) , with deep - ploughing consequences for the physics of elementary particles '' Julian Schwinger United States Shin'ichirō Tomonaga Japan 1966 Alfred Kastler France `` for the discovery and development of optical methods for studying Hertzian resonances in atoms '' 1967 Hans Albrecht Bethe United States `` for his contributions to the theory of nuclear reactions , especially his discoveries concerning the energy production in stars '' 1968 Luis Walter Alvarez United States `` for his decisive contributions to elementary particle physics , in particular the discovery of a large number of resonance states , made possible through his development of the technique of using hydrogen bubble chamber and data analysis '' 1969 Murray Gell - Mann United States `` for his contributions and discoveries concerning the classification of elementary particles and their interactions '' 1970 Hannes Olof Gösta Alfvén Sweden `` for fundamental work and discoveries in magneto - hydrodynamics with fruitful applications in different parts of plasma physics '' Louis Néel France `` for fundamental work and discoveries concerning antiferromagnetism and ferrimagnetism which have led to important applications in solid state physics '' 1971 Dennis Gabor Hungary United Kingdom `` for his invention and development of the holographic method '' 1972 John Bardeen United States `` for their jointly developed theory of superconductivity , usually called the BCS - theory '' Leon Neil Cooper United States John Robert Schrieffer United States 1973 Leo Esaki Japan `` for their experimental discoveries regarding tunneling phenomena in semiconductors and superconductors , respectively '' Ivar Giaever United States Norway Brian David Josephson United Kingdom `` for his theoretical predictions of the properties of a supercurrent through a tunnel barrier , in particular those phenomena which are generally known as the Josephson effect '' Martin Ryle United Kingdom `` for their pioneering research in radio astrophysics : Ryle for his observations and inventions , in particular of the aperture synthesis technique , and Hewish for his decisive role in the discovery of pulsars '' Antony Hewish United Kingdom Aage Bohr Denmark `` for the discovery of the connection between collective motion and particle motion in atomic nuclei and the development of the theory of the structure of the atomic nucleus based on this connection '' Ben Roy Mottelson Denmark Leo James Rainwater United States 1976 Burton Richter United States `` for their pioneering work in the discovery of a heavy elementary particle of a new kind '' Samuel Chao Chung Ting United States 1977 Philip Warren Anderson United States `` for their fundamental theoretical investigations of the electronic structure of magnetic and disordered systems '' Nevill Francis Mott United Kingdom John Hasbrouck Van Vleck United States 1978 Pyotr Leonidovich Kapitsa Soviet Union `` for his basic inventions and discoveries in the area of low - temperature physics '' Arno Allan Penzias United States `` for their discovery of cosmic microwave background radiation '' Robert Woodrow Wilson United States 1979 Sheldon Lee Glashow United States `` for their contributions to the theory of the unified weak and electromagnetic interaction between elementary particles , including , inter alia , the prediction of the weak neutral current '' Abdus Salam Pakistan Steven Weinberg United States 1980 James Watson Cronin United States `` for the discovery of violations of fundamental symmetry principles in the decay of neutral K - mesons '' Val Logsdon Fitch United States 1981 Nicolaas Bloembergen Netherlands United States `` for their contribution to the development of laser spectroscopy '' Arthur Leonard Schawlow United States Kai Manne Börje Siegbahn Sweden `` for his contribution to the development of high - resolution electron spectroscopy '' 1982 Kenneth G. Wilson United States `` for his theory for critical phenomena in connection with phase transitions '' Subrahmanyan Chandrasekhar India United States `` for his theoretical studies of the physical processes of importance to the structure and evolution of the stars '' William Alfred Fowler United States `` for his theoretical and experimental studies of the nuclear reactions of importance in the formation of the chemical elements in the universe '' 1984 Carlo Rubbia Italy `` for their decisive contributions to the large project , which led to the discovery of the field particles W and Z , communicators of weak interaction '' Simon van der Meer Netherlands 1985 Klaus von Klitzing West Germany `` for the discovery of the quantized Hall effect '' 1986 Ernst Ruska West Germany `` for his fundamental work in electron optics , and for the design of the first electron microscope '' Gerd Binnig West Germany `` for their design of the scanning tunneling microscope '' Heinrich Rohrer Switzerland Johannes Georg Bednorz West Germany `` for their important break - through in the discovery of superconductivity in ceramic materials '' Karl Alexander Müller Switzerland 1988 Leon Max Lederman United States `` for the neutrino beam method and the demonstration of the doublet structure of the leptons through the discovery of the muon neutrino '' Melvin Schwartz United States Jack Steinberger United States 1989 Norman Foster Ramsey United States `` for the invention of the separated oscillatory fields method and its use in the hydrogen maser and other atomic clocks '' Hans Georg Dehmelt United States Germany `` for the development of the ion trap technique '' Wolfgang Paul West Germany 1990 Jerome I. Friedman United States `` for their pioneering investigations concerning deep inelastic scattering of electrons on protons and bound neutrons , which have been of essential importance for the development of the quark model in particle physics '' Henry Way Kendall United States Richard E. Taylor Canada 1991 Pierre - Gilles de Gennes France `` for discovering that methods developed for studying order phenomena in simple systems can be generalized to more complex forms of matter , in particular to liquid crystals and polymers '' 1992 Georges Charpak France Poland `` for his invention and development of particle detectors , in particular the multiwire proportional chamber '' 1993 Russell Alan Hulse United States `` for the discovery of a new type of pulsar , a discovery that has opened up new possibilities for the study of gravitation '' Joseph Hooton Taylor , Jr . United States 1994 Bertram Brockhouse Canada `` for the development of neutron spectroscopy '' and `` for pioneering contributions to the development of neutron scattering techniques for studies of condensed matter '' Clifford Glenwood Shull United States `` for the development of the neutron diffraction technique '' and `` for pioneering contributions to the development of neutron scattering techniques for studies of condensed matter '' 1995 Martin Lewis Perl United States `` for the discovery of the tau lepton '' and `` for pioneering experimental contributions to lepton physics '' Frederick Reines United States `` for the detection of the neutrino '' and `` for pioneering experimental contributions to lepton physics '' David Morris Lee United States `` for their discovery of superfluidity in helium - 3 '' Douglas D. Osheroff United States Robert Coleman Richardson United States 1997 Steven Chu United States `` for development of methods to cool and trap atoms with laser light . '' Claude Cohen - Tannoudji France William Daniel Phillips United States 1998 Robert B. Laughlin United States `` for their discovery of a new form of quantum fluid with fractionally charged excitations '' Horst Ludwig Störmer Germany Daniel Chee Tsui Republic of China United States 1999 Gerard ' t Hooft Netherlands `` for elucidating the quantum structure of electroweak interactions in physics '' Martinus J.G. Veltman Netherlands 2000 Zhores Ivanovich Alferov Russia `` for developing semiconductor heterostructures used in high - speed - and optoelectronics '' Herbert Kroemer Germany Jack St. Clair Kilby United States `` for his part in the invention of the integrated circuit '' 2001 Eric Allin Cornell United States `` for the achievement of Bose -- Einstein condensation in dilute gases of alkali atoms , and for early fundamental studies of the properties of the condensates '' Carl Edwin Wieman United States Wolfgang Ketterle Germany 2002 Raymond Davis , Jr . United States `` for pioneering contributions to astrophysics , in particular for the detection of cosmic neutrinos '' Masatoshi Koshiba Japan Riccardo Giacconi Italy United States `` for pioneering contributions to astrophysics , which have led to the discovery of cosmic X-ray sources '' 2003 Alexei Alexeyevich Abrikosov Russia United States `` for pioneering contributions to the theory of superconductors and superfluids '' Vitaly Lazarevich Ginzburg Russia Anthony James Leggett United Kingdom United States David J. Gross United States `` for the discovery of asymptotic freedom in the theory of the strong interaction '' Hugh David Politzer United States Frank Wilczek United States 2005 Roy J. Glauber United States `` for his contribution to the quantum theory of optical coherence '' John L. Hall United States `` for their contributions to the development of laser - based precision spectroscopy , including the optical frequency comb technique '' Theodor W. Hänsch Germany 2006 John C. Mather United States `` for their discovery of the blackbody form and anisotropy of the cosmic microwave background radiation '' George F. Smoot United States 2007 Albert Fert France `` for the discovery of giant magnetoresistance '' Peter Grünberg Germany 2008 Makoto Kobayashi Japan `` for the discovery of the origin of the broken symmetry which predicts the existence of at least three families of quarks in nature '' Toshihide Maskawa Japan Yoichiro Nambu Japan United States `` for the discovery of the mechanism of spontaneous broken symmetry in subatomic physics '' 2009 Charles K. Kao Hong Kong United Kingdom United States `` for groundbreaking achievements concerning the transmission of light in fibers for optical communication '' Willard S. Boyle Canada United States `` for the invention of an imaging semiconductor circuit -- the CCD sensor '' George E. Smith United States Andre Geim Russia United Kingdom Netherlands `` for groundbreaking experiments regarding the two - dimensional material graphene '' Konstantin Novoselov Russia United Kingdom 2011 Saul Perlmutter United States `` for the discovery of the accelerating expansion of the Universe through observations of distant supernovae '' Brian P. Schmidt Australia United States Adam G. Riess United States 2012 Serge Haroche France `` for ground - breaking experimental methods that enable measuring and manipulation of individual quantum systems . '' David J. Wineland United States 2013 François Englert Belgium `` for the theoretical discovery of a mechanism that contributes to our understanding of the origin of mass of subatomic particles , and which recently was confirmed through the discovery of the predicted fundamental particle , by the ATLAS and CMS experiments at CERN 's Large Hadron Collider '' Peter Higgs United Kingdom 2014 Isamu Akasaki Japan `` for the invention of efficient blue light - emitting diodes which has enabled bright and energy - saving white light sources '' Hiroshi Amano Japan Shuji Nakamura Japan United States 2015 Takaaki Kajita Japan `` for the discovery of neutrino oscillations , which shows that neutrinos have mass '' Arthur B. McDonald Canada 2016 David J. Thouless United Kingdom United States `` for theoretical discoveries of topological phase transitions and topological phases of matter '' F. Duncan M. Haldane United Kingdom United States John M. Kosterlitz United Kingdom United States 2017 Rainer Weiss Germany United States `` for decisive contributions to the LIGO detector and the observation of gravitational waves '' Kip Thorne United States Barry Barish United States
  Fact 0.0: Wilhelm Conrad Röntgen of Germany won the 1901 Nobel Prize in Physics in recognition of the extraordinary services he rendered by discovering the remarkable rays later named after him.
  Fact 0.1: Hendrik Lorentz of the Netherlands won the 1902 Nobel Prize in Physics in recognition of the extraordinary service he rendered by his researches into the influence of magnetism upon radiation phenomena.
  Fact 0.2: Pieter Zeeman of the Netherlands shared the 1902 Nobel Prize in Physics.
  Fact 0.3: Antoine Henri Becquerel of France won the 1903 Nobel Prize in Physics for his discovery of spontaneous radioactivity.
  Fact 0.4: Pierre Curie of France shared the 1903 Nobel Prize in Physics for joint researches on the radiation phenomena discovered by Henri Becquerel.
  Fact 0.5: Maria Skłodowska-Curie, listed as from Poland and France, shared the 1903 Nobel Prize in Physics for joint researches on the radiation phenomena discovered by Henri Becquerel.
  Fact 0.6: Lord Rayleigh of the United Kingdom won the 1904 Nobel Prize in Physics for investigating the densities of the most important gases and discovering argon in connection with these studies.
  Fact 0.7: Philipp Eduard Anton von Lenard, listed as from Austria-Hungary and Germany, won the 1905 Nobel Prize in Physics for his work on cathode rays.
  Fact 0.8: Joseph John Thomson of the United Kingdom won the 1906 Nobel Prize in Physics for his theoretical and experimental investigations on the conduction of electricity by gases.
  Fact 0.9: Albert Abraham Michelson, listed as from the United States and Poland, won the 1907 Nobel Prize in Physics for his optical precision instruments and the spectroscopic and metrological investigations carried out with them.
  Fact 0.10: Gabriel Lippmann of France won the 1908 Nobel Prize in Physics for his method of reproducing colours photographically based on the phenomenon of interference.
  Fact 0.11: Guglielmo Marconi of Italy won the 1909 Nobel Prize in Physics for his contributions to the development of wireless telegraphy.
  Fact 0.12: Karl Ferdinand Braun of Germany shared the 1909 Nobel Prize in Physics for his contributions to the development of wireless telegraphy.
  Fact 0.13: Johannes Diderik van der Waals of the Netherlands won the 1910 Nobel Prize in Physics for his work on the equation of state for gases and liquids.
  Fact 0.14: Wilhelm Wien of Germany won the 1911 Nobel Prize in Physics for his discoveries regarding the laws governing the radiation of heat.
  Fact 0.15: Nils Gustaf Dalén of Sweden won the 1912 Nobel Prize in Physics for his invention of automatic valves designed for use with gas accumulators in lighthouses and buoys.
  Fact 0.16: Heike Kamerlingh-Onnes of the Netherlands won the 1913 Nobel Prize in Physics for his investigations on the properties of matter at low temperatures, which led among other things to the production of liquid helium.
  Fact 0.17: Max von Laue of Germany won the 1914 Nobel Prize in Physics for discovering the diffraction of X-rays by crystals, an important step in the development of X-ray spectroscopy.
  Fact 0.18: William Henry Bragg of the United Kingdom won the 1915 Nobel Prize in Physics for his services in the analysis of crystal structure by means of X-rays, an important step in the development of X-ray crystallography.
  Fact 0.19: William Lawrence Bragg, listed as from Australia and the United Kingdom, shared the 1915 Nobel Prize in Physics for services in the analysis of crystal structure by means of X-rays.
  Fact 0.20: No Nobel Prize in Physics was awarded in 1916 because of World War I.
  Fact 0.21: Charles Glover Barkla of the United Kingdom won the 1917 Nobel Prize in Physics for discovering the characteristic Röntgen radiation of the elements, another important step in the development of X-ray spectroscopy.
  Fact 0.22: Max Planck of Germany won the 1918 Nobel Prize in Physics for his services to the advancement of physics by discovering energy quanta.
  Fact 0.23: Johannes Stark of Germany won the 1919 Nobel Prize in Physics for discovering the Doppler effect in canal rays and the splitting of spectral lines in electric fields.
  Fact 0.24: Charles Édouard Guillaume of Switzerland won the 1920 Nobel Prize in Physics for his service to precision measurements in physics by discovering anomalies in nickel-steel alloys.
  Fact 0.25: Albert Einstein of Germany and Switzerland won the 1921 Nobel Prize in Physics for his services to theoretical physics, especially his discovery of the law of the photoelectric effect.
  Fact 0.26: Niels Bohr of Denmark won the 1922 Nobel Prize in Physics for his services in investigating the structure of atoms and the radiation emanating from them.
  Fact 0.27: Robert Andrews Millikan of the United States won the 1923 Nobel Prize in Physics for his work on the elementary charge of electricity and on the photoelectric effect.
  Fact 0.28: Manne Siegbahn of Sweden won the 1924 Nobel Prize in Physics for his discoveries and research in the field of X-ray spectroscopy.
  Fact 0.29: James Franck of Germany won the 1925 Nobel Prize in Physics for discovering the laws governing the impact of an electron upon an atom.
  Fact 0.30: Gustav Hertz of Germany shared the 1925 Nobel Prize in Physics for discovering the laws governing the impact of an electron upon an atom.
  Fact 0.31: Jean Baptiste Perrin of France won the 1926 Nobel Prize in Physics for his work on the discontinuous structure of matter, especially his discovery of sedimentation equilibrium.
  Fact 0.32: Arthur Holly Compton of the United States won the 1927 Nobel Prize in Physics for discovering the effect named after him.
  Fact 0.33: Charles Thomson Rees Wilson of the United Kingdom shared the 1927 Nobel Prize in Physics for his method of making the paths of electrically charged particles visible by condensation of vapour.
  Fact 0.34: Owen Willans Richardson of the United Kingdom won the 1928 Nobel Prize in Physics for his work on the thermionic phenomenon and especially for discovering the law named after him.
  Fact 0.35: Louis Victor Pierre Raymond, 7th Duc de Broglie of France won the 1929 Nobel Prize in Physics for discovering the wave nature of electrons.
  Fact 0.36: Chandrasekhara Venkata Raman of India won the 1930 Nobel Prize in Physics for his work on the scattering of light and for discovering the effect named after him.
  Fact 0.37: No Nobel Prize in Physics was awarded in 1931.
  Fact 0.38: Werner Heisenberg of Germany won the 1932 Nobel Prize in Physics for the creation of quantum mechanics, the application of which led, among other things, to the discovery of the allotropic forms of hydrogen.
  Fact 0.39: Erwin Schrödinger of Austria won the 1933 Nobel Prize in Physics for the discovery of new productive forms of atomic theory.
  Fact 0.40: Paul Dirac of the United Kingdom shared the 1933 Nobel Prize in Physics for the discovery of new productive forms of atomic theory.
  Fact 0.41: No Nobel Prize in Physics was awarded in 1934.
  Fact 0.42: James Chadwick of the United Kingdom won the 1935 Nobel Prize in Physics for discovering the neutron.
  Fact 0.43: Victor Francis Hess of Austria won the 1936 Nobel Prize in Physics for discovering cosmic radiation.
  Fact 0.44: Carl David Anderson of the United States shared the 1936 Nobel Prize in Physics for discovering the positron.
  Fact 0.45: Clinton Joseph Davisson of the United States won the 1937 Nobel Prize in Physics for the experimental discovery of the diffraction of electrons by crystals.
  Fact 0.46: George Paget Thomson of the United Kingdom shared the 1937 Nobel Prize in Physics for the experimental discovery of the diffraction of electrons by crystals.
  Fact 0.47: Enrico Fermi of Italy won the 1938 Nobel Prize in Physics for demonstrating the existence of new radioactive elements produced by neutron irradiation.
  Fact 0.48: Enrico Fermi's 1938 Nobel Prize in Physics also recognized his related discovery of nuclear reactions brought about by slow neutrons.
  Fact 0.49: Ernest Lawrence of the United States won the 1939 Nobel Prize in Physics for the invention and development of the cyclotron and results obtained with it, especially regarding artificial radioactive elements.
  Fact 0.50: No Nobel Prize in Physics was awarded in 1940, 1941, or 1942 because of World War II.
  Fact 0.51: Otto Stern of the United States won the 1943 Nobel Prize in Physics for his contribution to the development of the molecular ray method and his discovery of the magnetic moment of the proton.
  Fact 0.52: Isidor Isaac Rabi, listed as from the United States and Poland, won the 1944 Nobel Prize in Physics for his resonance method for recording the magnetic properties of atomic nuclei.
  Fact 0.53: Wolfgang Pauli of Austria won the 1945 Nobel Prize in Physics for discovering the Exclusion Principle, also called the Pauli principle.
  Fact 0.54: Percy Williams Bridgman of the United States won the 1946 Nobel Prize in Physics for inventing an apparatus to produce extremely high pressures and for discoveries he made there in high pressure physics.
  Fact 0.55: Edward Victor Appleton of the United Kingdom won the 1947 Nobel Prize in Physics for his investigations of the physics of the upper atmosphere, especially the discovery of the so-called Appleton layer.
  Fact 0.56: Patrick Maynard Stuart Blackett of the United Kingdom won the 1948 Nobel Prize in Physics for developing the Wilson cloud chamber method and his discoveries with it in nuclear physics and cosmic radiation.
  Fact 0.57: Hideki Yukawa of Japan won the 1949 Nobel Prize in Physics for predicting the existence of mesons on the basis of theoretical work on nuclear forces.
  Fact 0.58: Cecil Frank Powell of the United Kingdom won the 1950 Nobel Prize in Physics for developing the photographic method of studying nuclear processes and for discoveries regarding mesons made with this method.
  Fact 0.59: John Douglas Cockcroft of the United Kingdom shared the 1951 Nobel Prize in Physics for pioneer work on the transmutation of atomic nuclei by artificially accelerated atomic particles.
  Fact 0.60: Ernest Thomas Sinton Walton, listed as from Ireland, shared the 1951 Nobel Prize in Physics for pioneer work on the transmutation of atomic nuclei by artificially accelerated atomic particles.
  Fact 0.61: Felix Bloch, listed as from Switzerland and the United States, shared the 1952 Nobel Prize in Physics for developing new methods for nuclear magnetic precision measurements and discoveries in connection with them.
  Fact 0.62: Edward Mills Purcell of the United States shared the 1952 Nobel Prize in Physics for developing new methods for nuclear magnetic precision measurements and discoveries in connection with them.
  Fact 0.63: Frits Zernike of the Netherlands won the 1953 Nobel Prize in Physics for demonstrating the phase contrast method, especially for inventing the phase contrast microscope.
  Fact 0.64: Max Born of West Germany won the 1954 Nobel Prize in Physics for his fundamental research in quantum mechanics, especially his statistical interpretation of the wavefunction.
  Fact 0.65: Walther Bothe of West Germany shared the 1954 Nobel Prize in Physics for the coincidence method and his discoveries made with it.
  Fact 0.66: Willis Eugene Lamb of the United States won the 1955 Nobel Prize in Physics for discoveries concerning the fine structure of the hydrogen spectrum.
  Fact 0.67: Polykarp Kusch, listed as from the United States and Germany, shared the 1955 Nobel Prize in Physics for his precision determination of the magnetic moment of the electron.
  Fact 0.68: John Bardeen of the United States shared the 1956 Nobel Prize in Physics for researches on semiconductors and the discovery of the transistor effect.
  Fact 0.69: Walter Houser Brattain of the United States shared the 1956 Nobel Prize in Physics for researches on semiconductors and the discovery of the transistor effect.
  Fact 0.70: William Bradford Shockley of the United States shared the 1956 Nobel Prize in Physics for researches on semiconductors and the discovery of the transistor effect.
  Fact 0.71: Tsung-Dao Lee of the Republic of China won the 1957 Nobel Prize in Physics for a penetrating investigation of the so-called parity laws, which led to important discoveries regarding elementary particles.
  Fact 0.72: Chen Ning Yang of the Republic of China shared the 1957 Nobel Prize in Physics for a penetrating investigation of the so-called parity laws, which led to important discoveries regarding elementary particles.
  Fact 0.73: Pavel Alekseyevich Cherenkov of the Soviet Union won the 1958 Nobel Prize in Physics for the discovery and interpretation of the Cherenkov effect.
  Fact 0.74: Ilya Frank of the Soviet Union shared the 1958 Nobel Prize in Physics for the discovery and interpretation of the Cherenkov effect.
  Fact 0.75: Igor Yevgenyevich Tamm of the Soviet Union shared the 1958 Nobel Prize in Physics for the discovery and interpretation of the Cherenkov effect.
  Fact 0.76: Emilio Gino Segrè, listed as from Italy and the United States, shared the 1959 Nobel Prize in Physics for the discovery of the antiproton.
  Fact 0.77: Owen Chamberlain of the United States shared the 1959 Nobel Prize in Physics for the discovery of the antiproton.
  Fact 0.78: Donald Arthur Glaser of the United States won the 1960 Nobel Prize in Physics for the invention of the bubble chamber.
  Fact 0.79: Robert Hofstadter of the United States won the 1961 Nobel Prize in Physics for pioneering studies of electron scattering in atomic nuclei and for discoveries about the structure of the nucleons.
  Fact 0.80: Rudolf Ludwig Mössbauer of West Germany shared the 1961 Nobel Prize in Physics for his researches on the resonance absorption of gamma radiation and his discovery of the effect that bears his name.
  Fact 0.81: Lev Davidovich Landau of the Soviet Union won the 1962 Nobel Prize in Physics for his pioneering theories for condensed matter, especially liquid helium.
  Fact 0.82: Eugene Paul Wigner, listed as from Hungary and the United States, shared the 1963 Nobel Prize in Physics for his contributions to the theory of the atomic nucleus and elementary particles, particularly through the discovery and application of fundamental symmetry principles.
  Fact 0.83: Maria Goeppert-Mayer of the United States shared the 1963 Nobel Prize in Physics for discoveries concerning nuclear shell structure.
  Fact 0.84: J. Hans D. Jensen of West Germany shared the 1963 Nobel Prize in Physics for discoveries concerning nuclear shell structure.
  Fact 0.85: Nicolay Gennadiyevich Basov of the Soviet Union shared the 1964 Nobel Prize in Physics for fundamental work in quantum electronics that led to oscillators and amplifiers based on the maser-laser principle.
  Fact 0.86: Alexander Prokhorov of the Soviet Union shared the 1964 Nobel Prize in Physics for fundamental work in quantum electronics that led to oscillators and amplifiers based on the maser-laser principle.
  Fact 0.87: Charles Hard Townes of the United States shared the 1964 Nobel Prize in Physics for fundamental work in quantum electronics that led to oscillators and amplifiers based on the maser-laser principle.
  Fact 0.88: Richard Phillips Feynman of the United States shared the 1965 Nobel Prize in Physics for fundamental work in quantum electrodynamics (QED), with deep consequences for the physics of elementary particles.
  Fact 0.89: Julian Schwinger of the United States shared the 1965 Nobel Prize in Physics for fundamental work in quantum electrodynamics (QED), with deep consequences for the physics of elementary particles.
  Fact 0.90: Shin'ichirō Tomonaga of Japan shared the 1965 Nobel Prize in Physics for fundamental work in quantum electrodynamics (QED), with deep consequences for the physics of elementary particles.
  Fact 0.91: Alfred Kastler of France won the 1966 Nobel Prize in Physics for the discovery and development of optical methods for studying Hertzian resonances in atoms.
  Fact 0.92: Hans Albrecht Bethe of the United States won the 1967 Nobel Prize in Physics for his contributions to the theory of nuclear reactions, especially his discoveries concerning energy production in stars.
  Fact 0.93: Luis Walter Alvarez of the United States won the 1968 Nobel Prize in Physics for his decisive contributions to elementary particle physics, in particular the discovery of a large number of resonance states, made possible by his development of the hydrogen bubble chamber technique and data analysis.
  Fact 0.94: Murray Gell-Mann of the United States won the 1969 Nobel Prize in Physics for his contributions and discoveries concerning the classification of elementary particles and their interactions.
  Fact 0.95: Hannes Olof Gösta Alfvén of Sweden won the 1970 Nobel Prize in Physics for fundamental work and discoveries in magneto-hydrodynamics with fruitful applications in different parts of plasma physics.
  Fact 0.96: Louis Néel of France shared the 1970 Nobel Prize in Physics for fundamental work and discoveries concerning antiferromagnetism and ferrimagnetism, which led to important applications in solid state physics.
  Fact 0.97: Dennis Gabor, listed as from Hungary and the United Kingdom, won the 1971 Nobel Prize in Physics for his invention and development of the holographic method.
  Fact 0.98: John Bardeen of the United States shared the 1972 Nobel Prize in Physics for jointly developing the theory of superconductivity, usually called the BCS theory.
  Fact 0.99: Leon Neil Cooper of the United States shared the 1972 Nobel Prize in Physics for jointly developing the theory of superconductivity, usually called the BCS theory.
  Fact 0.100: John Robert Schrieffer of the United States shared the 1972 Nobel Prize in Physics for jointly developing the theory of superconductivity, usually called the BCS theory.
  Fact 0.101: Leo Esaki of Japan shared the 1973 Nobel Prize in Physics for experimental discoveries regarding tunneling phenomena in semiconductors and superconductors, respectively.
  Fact 0.102: Ivar Giaever, listed as from the United States and Norway, shared the 1973 Nobel Prize in Physics for experimental discoveries regarding tunneling phenomena in semiconductors and superconductors, respectively.
  Fact 0.103: Brian David Josephson of the United Kingdom shared the 1973 Nobel Prize in Physics for theoretical predictions of the properties of a supercurrent through a tunnel barrier, especially the phenomena generally known as the Josephson effect.
  Fact 0.104: Martin Ryle of the United Kingdom shared a Nobel Prize in Physics for pioneering research in radio astrophysics. Ryle was recognized for his observations and inventions, in particular of the aperture synthesis technique.
  Fact 0.105: Antony Hewish of the United Kingdom shared a Nobel Prize in Physics for pioneering research in radio astrophysics. Hewish was recognized for his decisive role in the discovery of pulsars.
  Fact 0.106: Aage Bohr of Denmark shared a Nobel Prize in Physics for discovering the connection between collective motion and particle motion in atomic nuclei and developing the theory of the structure of the atomic nucleus based on this connection.
  Fact 0.107: Ben Roy Mottelson of Denmark shared a Nobel Prize in Physics for discovering the connection between collective motion and particle motion in atomic nuclei and developing the theory of the structure of the atomic nucleus based on this connection.
  Fact 0.108: Leo James Rainwater of the United States shared a Nobel Prize in Physics for discovering the connection between collective motion and particle motion in atomic nuclei and developing the theory of the structure of the atomic nucleus based on this connection.
  Fact 0.109: Burton Richter of the United States shared the 1976 Nobel Prize in Physics for pioneering work in the discovery of a heavy elementary particle of a new kind.
  Fact 0.110: Samuel Chao Chung Ting of the United States shared the 1976 Nobel Prize in Physics for pioneering work in the discovery of a heavy elementary particle of a new kind.
  Fact 0.111: Philip Warren Anderson of the United States shared the 1977 Nobel Prize in Physics for fundamental theoretical investigations of the electronic structure of magnetic and disordered systems.
  Fact 0.112: Nevill Francis Mott of the United Kingdom shared the 1977 Nobel Prize in Physics for fundamental theoretical investigations of the electronic structure of magnetic and disordered systems.
  Fact 0.113: John Hasbrouck Van Vleck of the United States shared the 1977 Nobel Prize in Physics for fundamental theoretical investigations of the electronic structure of magnetic and disordered systems.
  Fact 0.114: Pyotr Leonidovich Kapitsa of the Soviet Union shared the 1978 Nobel Prize in Physics for his basic inventions and discoveries in the area of low-temperature physics.
  Fact 0.115: Arno Allan Penzias of the United States shared the 1978 Nobel Prize in Physics for discovering the cosmic microwave background radiation.
  Fact 0.116: Robert Woodrow Wilson of the United States shared the 1978 Nobel Prize in Physics for discovering the cosmic microwave background radiation.
  Fact 0.117: Sheldon Lee Glashow of the United States shared the 1979 Nobel Prize in Physics for contributions to the theory of the unified weak and electromagnetic interaction between elementary particles, including the prediction of the weak neutral current.
  Fact 0.118: Abdus Salam of Pakistan shared the 1979 Nobel Prize in Physics for contributions to the theory of the unified weak and electromagnetic interaction between elementary particles, including the prediction of the weak neutral current.
  Fact 0.119: Steven Weinberg of the United States shared the 1979 Nobel Prize in Physics for contributions to the theory of the unified weak and electromagnetic interaction between elementary particles, including the prediction of the weak neutral current.
  Fact 0.120: James Watson Cronin of the United States shared the 1980 Nobel Prize in Physics for discovering violations of fundamental symmetry principles in the decay of neutral K-mesons.
  Fact 0.121: Val Logsdon Fitch of the United States shared the 1980 Nobel Prize in Physics for discovering violations of fundamental symmetry principles in the decay of neutral K-mesons.
  Fact 0.122: Nicolaas Bloembergen, listed as from the Netherlands and the United States, shared the 1981 Nobel Prize in Physics for his contribution to the development of laser spectroscopy.
  Fact 0.123: Arthur Leonard Schawlow of the United States shared the 1981 Nobel Prize in Physics for his contribution to the development of laser spectroscopy.
  Fact 0.124: Kai Manne Börje Siegbahn of Sweden shared the 1981 Nobel Prize in Physics for his contribution to the development of high-resolution electron spectroscopy.
  Fact 0.125: Kenneth G. Wilson of the United States won the 1982 Nobel Prize in Physics for his theory of critical phenomena in connection with phase transitions.
  Fact 0.126: Subrahmanyan Chandrasekhar, listed as from India and the United States, shared the 1982 Nobel Prize in Physics for his theoretical studies of physical processes important to the structure and evolution of the stars.
  Fact 0.127: William Alfred Fowler of the United States shared the 1982 Nobel Prize in Physics for his theoretical and experimental studies of nuclear reactions important in the formation of the chemical elements in the universe.
  Fact 0.128: Carlo Rubbia of Italy shared the 1984 Nobel Prize in Physics for decisive contributions to the large project that led to the discovery of the field particles W and Z, communicators of the weak interaction.
  Fact 0.129: Simon van der Meer of the Netherlands shared the 1984 Nobel Prize in Physics for decisive contributions to the large project that led to the discovery of the field particles W and Z, communicators of the weak interaction.
  Fact 0.130: Klaus von Klitzing of West Germany won the 1985 Nobel Prize in Physics for the discovery of the quantized Hall effect.
  Fact 0.131: Ernst Ruska of West Germany shared the 1986 Nobel Prize in Physics for his fundamental work in electron optics and for the design of the first electron microscope.
  Fact 0.132: Gerd Binnig of West Germany shared the 1986 Nobel Prize in Physics for designing the scanning tunneling microscope.
  Fact 0.133: Heinrich Rohrer of Switzerland shared the 1986 Nobel Prize in Physics for designing the scanning tunneling microscope.
  Fact 0.134: Johannes Georg Bednorz of West Germany shared the 1986 Nobel Prize in Physics for an important break-through in the discovery of superconductivity in ceramic materials.
  Fact 0.135: Karl Alexander Müller of Switzerland shared the 1986 Nobel Prize in Physics for an important break-through in the discovery of superconductivity in ceramic materials.
  Fact 0.136: Leon Max Lederman of the United States shared the 1988 Nobel Prize in Physics for the neutrino beam method and the demonstration of the doublet structure of the leptons through the discovery of the muon neutrino.
  Fact 0.137: Melvin Schwartz of the United States shared the 1988 Nobel Prize in Physics for the neutrino beam method and the demonstration of the doublet structure of the leptons through the discovery of the muon neutrino.
  Fact 0.138: Jack Steinberger of the United States shared the 1988 Nobel Prize in Physics for the neutrino beam method and the demonstration of the doublet structure of the leptons through the discovery of the muon neutrino.
  Fact 0.139: Norman Foster Ramsey of the United States shared the 1989 Nobel Prize in Physics for inventing the separated oscillatory fields method and its use in the hydrogen maser and other atomic clocks.
  Fact 0.140: Hans Georg Dehmelt, listed as from the United States and Germany, shared the 1989 Nobel Prize in Physics for developing the ion trap technique.
  Fact 0.141: Wolfgang Paul of West Germany shared the 1989 Nobel Prize in Physics for developing the ion trap technique.
  Fact 0.142: Jerome I. Friedman of the United States shared the 1990 Nobel Prize in Physics for pioneering investigations of deep inelastic scattering of electrons on protons and bound neutrons, which were essential for the development of the quark model in particle physics.
  Fact 0.143: Henry Way Kendall of the United States shared the 1990 Nobel Prize in Physics for pioneering investigations of deep inelastic scattering of electrons on protons and bound neutrons, which were essential for the development of the quark model in particle physics.
  Fact 0.144: Richard E. Taylor of Canada shared the 1990 Nobel Prize in Physics for pioneering investigations of deep inelastic scattering of electrons on protons and bound neutrons, which were essential for the development of the quark model in particle physics.
  Fact 0.145: Pierre-Gilles de Gennes of France won the 1991 Nobel Prize in Physics for discovering that methods developed for studying order phenomena in simple systems can be generalized to more complex forms of matter, in particular to liquid crystals and polymers.
  Fact 0.146: Georges Charpak, listed as from France and Poland, won the 1992 Nobel Prize in Physics for his invention and development of particle detectors, in particular the multiwire proportional chamber.
  Fact 0.147: Russell Alan Hulse of the United States shared the 1993 Nobel Prize in Physics for discovering a new type of pulsar, a discovery that opened up new possibilities for studying gravitation.
  Fact 0.148: Joseph Hooton Taylor Jr. of the United States shared the 1993 Nobel Prize in Physics for discovering a new type of pulsar, a discovery that opened up new possibilities for studying gravitation.
  Fact 0.149: Bertram Brockhouse of Canada shared the 1994 Nobel Prize in Physics for the development of neutron spectroscopy and for pioneering contributions to neutron scattering techniques for studies of condensed matter.
  Fact 0.150: Clifford Glenwood Shull of the United States shared the 1994 Nobel Prize in Physics for the development of the neutron diffraction technique and for pioneering contributions to neutron scattering techniques for studies of condensed matter.
  Fact 0.151: Martin Lewis Perl of the United States shared the 1995 Nobel Prize in Physics for discovering the tau lepton and for pioneering experimental contributions to lepton physics.
  Fact 0.152: Frederick Reines of the United States shared the 1995 Nobel Prize in Physics for detecting the neutrino and for pioneering experimental contributions to lepton physics.
  Fact 0.153: David Morris Lee of the United States shared a Nobel Prize in Physics for discovering superfluidity in helium-3.
  Fact 0.154: Douglas D. Osheroff of the United States shared a Nobel Prize in Physics for discovering superfluidity in helium-3.
  Fact 0.155: Robert Coleman Richardson of the United States shared a Nobel Prize in Physics for discovering superfluidity in helium-3.
  Fact 0.156: Steven Chu of the United States won the 1997 Nobel Prize in Physics for developing methods to cool and trap atoms with laser light.
  Fact 0.157: Claude Cohen-Tannoudji of France shared the 1997 Nobel Prize in Physics for developing methods to cool and trap atoms with laser light.
  Fact 0.158: William Daniel Phillips of the United States shared the 1997 Nobel Prize in Physics for developing methods to cool and trap atoms with laser light.
  Fact 0.159: Robert B. Laughlin of the United States shared the 1998 Nobel Prize in Physics for discovering a new form of quantum fluid with fractionally charged excitations.
  Fact 0.160: Horst Ludwig Störmer of Germany shared the 1998 Nobel Prize in Physics for discovering a new form of quantum fluid with fractionally charged excitations.
  Fact 0.161: Daniel Chee Tsui, listed as from the Republic of China and the United States, shared the 1998 Nobel Prize in Physics for discovering a new form of quantum fluid with fractionally charged excitations.
  Fact 0.162: Gerard 't Hooft of the Netherlands shared the 1999 Nobel Prize in Physics for elucidating the quantum structure of electroweak interactions in physics.
  Fact 0.163: Martinus J.G. Veltman of the Netherlands shared the 1999 Nobel Prize in Physics for elucidating the quantum structure of electroweak interactions in physics.
  Fact 0.164: Zhores Ivanovich Alferov of Russia shared the 2000 Nobel Prize in Physics for developing semiconductor heterostructures used in high-speed and optoelectronics.
  Fact 0.165: Herbert Kroemer of Germany shared the 2000 Nobel Prize in Physics for developing semiconductor heterostructures used in high-speed and optoelectronics.
  Fact 0.166: Jack St. Clair Kilby of the United States shared the 2000 Nobel Prize in Physics for his part in the invention of the integrated circuit.
  Fact 0.167: Eric Allin Cornell of the United States shared the 2001 Nobel Prize in Physics for achieving Bose-Einstein condensation in dilute gases of alkali atoms and for early fundamental studies of the properties of the condensates.
  Fact 0.168: Carl Edwin Wieman of the United States shared the 2001 Nobel Prize in Physics for achieving Bose-Einstein condensation in dilute gases of alkali atoms and for early fundamental studies of the properties of the condensates.
  Fact 0.169: Wolfgang Ketterle of Germany shared the 2001 Nobel Prize in Physics for achieving Bose-Einstein condensation in dilute gases of alkali atoms and for early fundamental studies of the properties of the condensates.
  Fact 0.170: Raymond Davis Jr. of the United States shared the 2002 Nobel Prize in Physics for pioneering contributions to astrophysics, in particular the detection of cosmic neutrinos.
  Fact 0.171: Masatoshi Koshiba of Japan shared the 2002 Nobel Prize in Physics for pioneering contributions to astrophysics, in particular the detection of cosmic neutrinos.
  Fact 0.172: Riccardo Giacconi, listed as from Italy and the United States, shared the 2002 Nobel Prize in Physics for pioneering contributions to astrophysics that led to the discovery of cosmic X-ray sources.
  Fact 0.173: Alexei Alexeyevich Abrikosov, listed as from Russia and the United States, shared the 2003 Nobel Prize in Physics for pioneering contributions to the theory of superconductors and superfluids.
  Fact 0.174: Vitaly Lazarevich Ginzburg of Russia shared the 2003 Nobel Prize in Physics for pioneering contributions to the theory of superconductors and superfluids.
  Fact 0.175: Anthony James Leggett, listed as from the United Kingdom and the United States, shared the 2003 Nobel Prize in Physics for pioneering contributions to the theory of superconductors and superfluids.
  Fact 0.176: David J. Gross of the United States shared a Nobel Prize in Physics for discovering asymptotic freedom in the theory of the strong interaction.
  Fact 0.177: Hugh David Politzer of the United States shared a Nobel Prize in Physics for discovering asymptotic freedom in the theory of the strong interaction.
  Fact 0.178: Frank Wilczek of the United States shared a Nobel Prize in Physics for discovering asymptotic freedom in the theory of the strong interaction.
  Fact 0.179: Roy J. Glauber of the United States shared the 2005 Nobel Prize in Physics for his contribution to the quantum theory of optical coherence.
  Fact 0.180: John L. Hall of the United States shared the 2005 Nobel Prize in Physics for contributions to the development of laser-based precision spectroscopy, including the optical frequency comb technique.
  Fact 0.181: Theodor W. Hänsch of Germany shared the 2005 Nobel Prize in Physics for contributions to the development of laser-based precision spectroscopy, including the optical frequency comb technique.
  Fact 0.182: John C. Mather of the United States shared the 2006 Nobel Prize in Physics for discovering the blackbody form and anisotropy of the cosmic microwave background radiation.
  Fact 0.183: George F. Smoot of the United States shared the 2006 Nobel Prize in Physics for discovering the blackbody form and anisotropy of the cosmic microwave background radiation.
  Fact 0.184: Albert Fert of France shared the 2007 Nobel Prize in Physics for discovering giant magnetoresistance.
  Fact 0.185: Peter Grünberg of Germany shared the 2007 Nobel Prize in Physics for discovering giant magnetoresistance.
  Fact 0.186: Makoto Kobayashi of Japan shared the 2008 Nobel Prize in Physics for discovering the origin of the broken symmetry which predicts the existence of at least three families of quarks in nature.
  Fact 0.187: Toshihide Maskawa of Japan shared the 2008 Nobel Prize in Physics for discovering the origin of the broken symmetry which predicts the existence of at least three families of quarks in nature.
  Fact 0.188: Yoichiro Nambu, listed as from Japan and the United States, shared the 2008 Nobel Prize in Physics for discovering the mechanism of spontaneous broken symmetry in subatomic physics.
  Fact 0.189: Charles K. Kao, listed as from Hong Kong, the United Kingdom, and the United States, shared the 2009 Nobel Prize in Physics for groundbreaking achievements concerning the transmission of light in fibers for optical communication.
  Fact 0.190: Willard S. Boyle, listed as from Canada and the United States, shared the 2009 Nobel Prize in Physics for the invention of an imaging semiconductor circuit, the CCD sensor.
  Fact 0.191: George E. Smith of the United States shared the 2009 Nobel Prize in Physics for the invention of an imaging semiconductor circuit, the CCD sensor.
  Fact 0.192: Andre Geim, listed as from Russia, the United Kingdom, and the Netherlands, shared a Nobel Prize in Physics for groundbreaking experiments regarding the two-dimensional material graphene.
  Fact 0.193: Konstantin Novoselov, listed as from Russia and the United Kingdom, shared a Nobel Prize in Physics for groundbreaking experiments regarding the two-dimensional material graphene.
  Fact 0.194: Saul Perlmutter of the United States shared the 2011 Nobel Prize in Physics for discovering the accelerating expansion of the Universe through observations of distant supernovae.
  Fact 0.195: Brian P. Schmidt, listed as from Australia and the United States, shared the 2011 Nobel Prize in Physics for discovering the accelerating expansion of the Universe through observations of distant supernovae.
  Fact 0.196: Adam G. Riess of the United States shared the 2011 Nobel Prize in Physics for discovering the accelerating expansion of the Universe through observations of distant supernovae.
  Fact 0.197: Serge Haroche of France shared the 2012 Nobel Prize in Physics for ground-breaking experimental methods that enable measuring and manipulation of individual quantum systems.
  Fact 0.198: David J. Wineland of the United States shared the 2012 Nobel Prize in Physics for ground-breaking experimental methods that enable measuring and manipulation of individual quantum systems.
  Fact 0.199: François Englert of Belgium shared the 2013 Nobel Prize in Physics for the theoretical discovery of a mechanism that contributes to understanding the origin of mass of subatomic particles.
  Fact 0.200: Peter Higgs of the United Kingdom shared the 2013 Nobel Prize in Physics for the theoretical discovery of a mechanism that contributes to understanding the origin of mass of subatomic particles.
  Fact 0.201: The mechanism for the origin of mass of subatomic particles was confirmed through the discovery of the predicted fundamental particle by the ATLAS and CMS experiments at CERN's Large Hadron Collider.
  Fact 0.202: Isamu Akasaki of Japan shared the 2014 Nobel Prize in Physics for the invention of efficient blue light-emitting diodes, which enabled bright and energy-saving white light sources.
  Fact 0.203: Hiroshi Amano of Japan shared the 2014 Nobel Prize in Physics for the invention of efficient blue light-emitting diodes, which enabled bright and energy-saving white light sources.
  Fact 0.204: Shuji Nakamura, listed as from Japan and the United States, shared the 2014 Nobel Prize in Physics for the invention of efficient blue light-emitting diodes, which enabled bright and energy-saving white light sources.
  Fact 0.205: Takaaki Kajita of Japan shared the 2015 Nobel Prize in Physics for the discovery of neutrino oscillations, which shows that neutrinos have mass.
  Fact 0.206: Arthur B. McDonald of Canada shared the 2015 Nobel Prize in Physics for the discovery of neutrino oscillations, which shows that neutrinos have mass.
  Fact 0.207: David J. Thouless, listed as from the United Kingdom and the United States, shared the 2016 Nobel Prize in Physics for theoretical discoveries of topological phase transitions and topological phases of matter.
  Fact 0.208: F. Duncan M. Haldane, listed as from the United Kingdom and the United States, shared the 2016 Nobel Prize in Physics for theoretical discoveries of topological phase transitions and topological phases of matter.
  Fact 0.209: John M. Kosterlitz, listed as from the United Kingdom and the United States, shared the 2016 Nobel Prize in Physics for theoretical discoveries of topological phase transitions and topological phases of matter.
  Fact 0.210: Rainer Weiss, listed as from Germany and the United States, shared the 2017 Nobel Prize in Physics for decisive contributions to the LIGO detector and the observation of gravitational waves.
  Fact 0.211: Kip Thorne of the United States shared the 2017 Nobel Prize in Physics for decisive contributions to the LIGO detector and the observation of gravitational waves.
  Fact 0.212: Barry Barish of the United States shared the 2017 Nobel Prize in Physics for decisive contributions to the LIGO detector and the observation of gravitational waves.
Sentence 1: The first Nobel Prize in Physics was awarded in 1901 to Wilhelm Conrad Röntgen , of Germany , who received 150,782 SEK , which is equal to 7,731,004 SEK in December 2007 . John Bardeen is the only laureate to win the prize twice -- in 1956 and 1972 . Maria Skłodowska - Curie also won two Nobel Prizes , for physics in 1903 and chemistry in 1911 . William Lawrence Bragg was , until October 2014 , the youngest ever Nobel laureate ; he won the prize in 1915 at the age of 25 . Two women have won the prize : Curie and Maria Goeppert - Mayer ( 1963 ) . As of 2017 , the prize has been awarded to 206 individuals . There have been six years in which the Nobel Prize in Physics was not awarded ( 1916 , 1931 , 1934 , 1940 -- 1942 ) .
  Fact 1.0: The first Nobel Prize in Physics was awarded in 1901 to Wilhelm Conrad Röntgen, of Germany.
  Fact 1.1: Röntgen received 150,782 SEK for the first Nobel Prize in Physics.
  Fact 1.2: 150,782 SEK is equal to 7,731,004 SEK in December 2007.
  Fact 1.3: John Bardeen is the only laureate to win the Nobel Prize in Physics twice.
  Fact 1.4: John Bardeen won the Nobel Prize in Physics in 1956 and in 1972.
  Fact 1.5: Maria Skłodowska-Curie also won two Nobel Prizes, one for physics in 1903 and one for chemistry in 1911.
  Fact 1.6: William Lawrence Bragg was the youngest ever Nobel laureate until October 2014.
  Fact 1.7: William Lawrence Bragg won the prize in 1915 at the age of 25.
  Fact 1.8: Two women have won the Nobel Prize in Physics: Maria Skłodowska-Curie and Maria Goeppert-Mayer.
  Fact 1.9: Maria Goeppert-Mayer won the Nobel Prize in Physics in 1963.
  Fact 1.10: As of 2017, the Nobel Prize in Physics has been awarded to 206 individuals.
  Fact 1.11: There have been six years in which the Nobel Prize in Physics was not awarded: 1916, 1931, 1934, and 1940 to 1942.

Retrieved items:
<item 1>
List of Nobel laureates in Physics
Jump up ^ `` The Nobel Prize in Physics 1925 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1926 '' . Nobel
Foundation . Retrieved 2008 - 10 - 09 . ^ Jump up to : `` The Nobel Prize in Physics 1927 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The
Nobel Prize in Physics 1928 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1929 '' . Nobel Foundation . Retrieved
2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1930 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1932 ''
. Nobel Foundation . Retrieved 2008 - 10 - 09 .
</item 1>

<item 2>
List of Nobel laureates in Physics
Jump up ^ `` The Nobel Prize in Physics 1918 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1919 '' . Nobel
Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1920 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel
Prize in Physics 1921 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1922 '' . Nobel Foundation . Retrieved 2008 -
10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1923 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1924 '' .
Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1925 '' .
</item 2>

<item 3>
List of Nobel laureates in Physics
Jump up ^ `` The Nobel Prize in Physics 1943 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1944 '' . Nobel
Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1945 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel
Prize in Physics 1946 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1947 '' . Nobel Foundation . Retrieved 2008 -
10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1948 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1949 '' .
Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1950 '' .
</item 3>

<item 4>
List of Nobel laureates in Physics
Smoot United States 2007 Albert Fert France `` for the discovery of giant magnetoresistance '' Peter Grünberg Germany 2008 Makoto Kobayashi Japan `` for the
discovery of the origin of the broken symmetry which predicts the existence of at least three families of quarks in nature '' Toshihide Maskawa Japan Yoichiro
Nambu Japan United States `` for the discovery of the mechanism of spontaneous broken symmetry in subatomic physics '' 2009 Charles K. Kao Hong Kong United
Kingdom United States `` for groundbreaking achievements concerning the transmission of light in fibers for optical communication '' Willard S. Boyle Canada
United States `` for the invention of an imaging semiconductor circuit -- the CCD sensor '' George E. Smith United States Andre Geim Russia United Kingdom
Netherlands `` for groundbreaking experiments regarding the two - dimensional material graphene '' Konstantin Novoselov Russia United Kingdom 2011 Saul
Perlmutter United States `` for the discovery of the accelerating expansion of the Universe through observations of distant supernovae '' Brian P. Schmidt
Australia United States Adam G. Riess United States 2012 Serge Haroche France `` for ground - breaking experimental methods that enable measuring and
manipulation of individual quantum systems . ''
</item 4>

<item 5>
List of Nobel laureates in Physics
As of 2017 , the prize has been awarded to 206 individuals . There have been six years in which the Nobel Prize in Physics was not awarded ( 1916 , 1931 , 1934
, 1940 -- 1942 ) . Laureates ( edit ) Year Laureate Country Rationale 1901 Wilhelm Conrad Röntgen Germany `` in recognition of the extraordinary services he has
rendered by the discovery of the remarkable rays subsequently named after him '' 1902 Hendrik Lorentz Netherlands `` in recognition of the extraordinary service
they rendered by their researches into the influence of magnetism upon radiation phenomena '' Pieter Zeeman Netherlands 1903 Antoine Henri Becquerel France ``
for his discovery of spontaneous radioactivity '' Pierre Curie France `` for their joint researches on the radiation phenomena discovered by Professor Henri
Becquerel '' Maria Skłodowska - Curie Poland France 1904 Lord Rayleigh United Kingdom `` for his investigations of the densities of the most important gases and
for his discovery of argon in connection with these studies '' 1905 Philipp Eduard Anton von Lenard Austria - Hungary Germany `` for his work on cathode rays ''
1906 Joseph
</item 5>

<item 6>
List of Nobel laureates in Physics
Jump up ^ `` The Nobel Prize in Physics 1910 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1911 '' . Nobel
Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1912 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel
Prize in Physics 1913 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1914 '' . Nobel Foundation . Retrieved 2008 -
10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1915 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1917 '' .
Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1918 '' .
</item 6>

<item 7>
List of Nobel laureates in Physics
Contents ( hide ) 1 Statistics 2 Laureates 3 See also 4 References 4.1 Notes 4.2 Citations 4.3 Sources 5 External links Statistics ( edit ) The first Nobel
Prize in Physics was awarded in 1901 to Wilhelm Conrad Röntgen , of Germany , who received 150,782 SEK , which is equal to 7,731,004 SEK in December 2007 . John
Bardeen is the only laureate to win the prize twice -- in 1956 and 1972 . Maria Skłodowska - Curie also won two Nobel Prizes , for physics in 1903 and chemistry
in 1911 . William Lawrence Bragg was , until October 2014 , the youngest ever Nobel laureate ; he won the prize in 1915 at the age of 25 . Two women have won
the prize : Curie and Maria Goeppert - Mayer ( 1963 ) . As of 2017 , the prize has been awarded to 206 individuals .
</item 7>

<item 8>
List of Nobel laureates in Physics
Yang / T.D. Lee 1958 Cherenkov / Frank / Tamm 1959 Segrè / Chamberlain 1960 Glaser 1961 Hofstadter / Mössbauer 1962 Landau 1963 Wigner / Goeppert - Mayer /
Jensen 1964 Townes / Basov / Prokhorov 1965 Tomonaga / Schwinger / Feynman 1966 Kastler 1967 Bethe 1968 Alvarez 1969 Gell - Mann 1970 Alfvén / Néel 1971 Gabor
1972 Bardeen / Cooper / Schrieffer 1973 Esaki / Giaever / Josephson 1974 Ryle / Hewish 1975 A. Bohr / Mottelson / Rainwater 1976 -- 2000 1976 Richter / Ting
1977 P.W. Anderson / Mott / Van Vleck 1978 Kapitsa / Penzias / R. Wilson 1979 Glashow / Salam / Weinberg 1980 Cronin / Fitch 1981 Bloembergen / Schawlow / K.
</item 8>

<item 9>
List of Nobel laureates in Physics
Jump up ^ `` The Nobel Prize in Physics 1950 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1951 '' . Nobel
Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1952 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel
Prize in Physics 1953 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . ^ Jump up to : `` The Nobel Prize in Physics 1954 '' . Nobel Foundation . Retrieved
2008 - 10 - 09 . ^ Jump up to : `` The Nobel Prize in Physics 1955 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics
1956 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 .
</item 9>

<item 10>
List of Nobel laureates in Physics
Claude Cohen - Tannoudji France William Daniel Phillips United States 1998 Robert B. Laughlin United States `` for their discovery of a new form of quantum
fluid with fractionally charged excitations '' Horst Ludwig Störmer Germany Daniel Chee Tsui Republic of China United States 1999 Gerard ' t Hooft Netherlands
`` for elucidating the quantum structure of electroweak interactions in physics '' Martinus J.G. Veltman Netherlands 2000 Zhores Ivanovich Alferov Russia `` for
developing semiconductor heterostructures used in high - speed - and optoelectronics '' Herbert Kroemer Germany Jack St. Clair Kilby United States `` for his
part in the invention of the integrated circuit '' 2001 Eric Allin Cornell United States `` for the achievement of Bose -- Einstein condensation in dilute gases
of alkali atoms , and for early fundamental studies of the properties of the condensates '' Carl Edwin Wieman United States Wolfgang Ketterle Germany 2002
Raymond Davis , Jr . United States `` for pioneering contributions to astrophysics ,
</item 10>
=== END ITEM id=832a1d27bc6eeb3a ===

=== ITEM id=c022ea0374f2e01b ===
Gold sentences and facts:
Sentence 0: Deadpool 2 is scheduled to be released in the United States on May 18 , 2018 . A sequel , Deadpool 3 , is in development .
  Fact 0.0: Deadpool 2 is scheduled to be released in the United States on May 18, 2018.
  Fact 0.1: Deadpool 3 is a sequel and is in development.

Retrieved items:
<item 1>
Deadpool 2
In January 2018 , the film 's release was moved up to May 18 , 2018 . A month later , Terry Crews was revealed to have a role in the film , the character
Shatterstar was confirmed to be appearing , and the production returned to Vancouver for six days of reshoots under a new working title , Daisy . While some
reports emerged by mid-March claiming that the reshoots were due to poor audience responses during test screenings of the film , and consisted of sweeping
changes , the film was soon confirmed to be testing better than the original did , up to 98 out of 100 over three different tests ( compared to a maximum score
of 91 during tests of the first film ) . Because of this , the reshoots were adding more of the elements that audiences responded positively to , including
additional material featuring Cable and Domino . By the end of the month , two hours - worth of filming had taken place in Los Angeles for a then - secret cameo
appearance , and the film 's first full trailer was released .
</item 1>

<item 2>
Deadpool 2
Cable 's inclusion in the potential sequel was confirmed by the character Deadpool , breaking the fourth wall , in the post-credit scene of the first film .
Domino , a character with connections to Cable in the comics , was also believed to be featured in the sequel . By the first film 's release , 20th Century Fox
had green - lit a sequel , with writers Rhett Reese and Paul Wernick returning to write the screenplay . Though director Tim Miller and producer / star Ryan
Reynolds were not confirmed for the sequel at the time , Fox was `` intent on keeping the creative team together '' . Miller and Reynolds ' involvement was
confirmed at the 2016 CinemaCon that April , though Miller had still not formally signed on to direct the sequel yet . He began work developing the script with
the writers , while Reynolds had signed a new contract granting him `` casting approval and other creative controls '' . It has to tonally and stylistically be
as fresh and original ( as the first film ) . That 's a big challenge especially because they had 10 years to gestate on the first movie and we do n't have that
kind of time on the second movie .
</item 2>

<item 3>
Deadpool 2
Reynolds , a fan of Leitch 's John Wick , said the director `` really understands those Deadpool sensibilities and where we need to take the franchise from here
. '' He added that Leitch `` can make a movie on an ultra tight minimal budget look like it was shot for 10 -- 15 times what it cost . '' Pre-production Reese
and Wernick had completed multiple drafts of the script by January 2017 , saying , `` It 's taken different twists and turns , but it 's really coalescing '' ;
the film was still on track to begin filming that year . The pair felt a responsibility to explore the team X-Force , which includes Deadpool , Cable , and
Domino in the comics , with Reese saying the sequel 's purpose `` is not to set up X-Force ( but ) it will likely set up X-Force . '' He did clarify that though
the film will be `` populated with a lot of characters ... it is still Deadpool 's movie '' .
</item 3>

<item 4>
Deadpool 2
As part of a promotion for the film with beverage company Mike 's Harder , Sister Margaret 's School for Wayward Girls -- the fictional bar featured in the
films -- will be emulated in pop - ups at the Alligator Lounge in New York from April 26 -- 28 and the Slipper Clutch in Los Angeles from May 10 -- 12 . The pop
- ups will respectively serve pizza and chimichangas , traditionally Deadpool 's favorite food , as well as Deadpool - inspired Harder drinks . The campaign
also includes the chance to win a trip to the film 's premiere through Harder , and the Los Angeles pop - up will benefit the nonprofit DTLA Film Festival . Box
office projection Initial box office projections for the film in March 2018 expected it to gross $100 million over its opening weekend in the United States and
Canada , with a final domestic total expected to be lower than that of the first film ( $363 million ) at around $242 million . On April 20 , 2018 , both
Fandango and Regal Cinemas announced that Deadpool 2 was the best pre-selling R - rated film in their respective histories .
</item 4>

<item 5>
Deadpool 2
Officially titling the film Deadpool 2 , the trailer also revealed that Crews was portraying Bedlam ; that Delaney was in the film , portraying Peter ; and that
Bill Skarsgård also had a role . The film 's final trailer , released a month later , confirmed that Lewis Tan would portray Shatterstar , joining Domino ,
Bedlam , and Peter as members of the X-Force in the film . Music Deadpool composer Junkie XL chose not to return for the sequel due to Miller 's departure . In
October 2017 , Tyler Bates was hired to write the score for the film . Release Deadpool 2 is set to be released in the United States on May 18 , 2018 . It was
previously scheduled for release on June 1 of that year . Marketing For the Fox presentation at CineEurope 2017 in June , Reynolds made a video message
featuring himself in costume as Deadpool from the film 's set . The first teaser poster , which pays homage to Norman Rockwell 's 1943 painting Freedom from
Want , was released that November .
</item 5>

<item 6>
Deadpool 2
The film is directed by David Leitch from a script by Rhett Reese , Paul Wernick , and Ryan Reynolds , with Reynolds starring in the title role alongside Josh
Brolin , Morena Baccarin , Julian Dennison , Zazie Beetz , T.J. Miller , Brianna Hildebrand , Jack Kesy , and Stefan Kapičić . In Deadpool 2 , Deadpool forms
the team X-Force to protect a young mutant from Cable . Plans for a sequel to Deadpool began before that film 's release , and were confirmed in February 2016 .
Though the original creative team of Reynolds , Reese , Wernick , and director Tim Miller were quickly set to return for the second film , Miller left the
project in October 2016 due to creative differences with Reynolds , and was soon replaced by Leitch . An extensive casting search took place to fill the role of
Cable , with Brolin ultimately cast ; the casting of Beetz as Domino was also noteworthy . Filming took place in British Columbia , Canada , from June to
October 2017 . During filming , stunt woman Joi `` SJ '' Harris died in a motorcycle accident .
</item 6>

<item 7>
Deadpool 2
Deadpool 2 - wikipedia Deadpool 2 Jump to : navigation , search Deadpool 2 Teaser poster Directed by David Leitch Produced by Simon Kinberg Ryan Reynolds Lauren
Shuler Donner Written by Rhett Reese Paul Wernick Ryan Reynolds Based on Deadpool by Rob Liefeld Fabian Nicieza Starring Ryan Reynolds Josh Brolin Morena
Baccarin Julian Dennison Zazie Beetz T.J. Miller Brianna Hildebrand Jack Kesy Stefan Kapičić Music by Tyler Bates Cinematography Jonathan Sela Edited by Craig
Alpert Elísabet Ronaldsdóttir Production company Marvel Entertainment The Donners ' Company Genre Films Distributed by 20th Century Fox Release date May 18 ,
2018 ( 2018 - 05 - 18 ) ( United States ) Country United States Language English Deadpool 2 is an upcoming American superhero film based on the Marvel Comics
character Deadpool , distributed by 20th Century Fox . It is intended to be the eleventh installment in the X-Men film series , and a sequel to the 2016 film
Deadpool .
</item 7>

<item 8>
Deadpool 2
Retrieved December 16 , 2017 . External links Official website Deadpool 2 on IMDb Deadpool Fabian Nicieza Rob Liefeld Supporting characters Agent X Avengers
Blind Al Bob , Agent of Hydra Sandi Brandenberg Cable Copycat Death Domino Emrys Killebrew Negasonic Teenage Warhead Outlaw Shiklah Siryn Spider - Man Weasel
Wolverine X-Men Antagonists Ajax Black Box Black Swan Black Talon Black Tom Cassidy Bullseye Doctor Bong Dracula Hit - Monkey Flag - Smasher Juggernaut Madcap
Sabretooth Slayback Taskmaster T - Ray Thanos Trapster Typhoid Mary Publications Cable & Deadpool Mrs. Deadpool and the Howling Commandos Wolverine and Deadpool
In other media Deadpool ( video game ) Deadpool ( film ) accolades box office records soundtrack No Good Deed Deadpool 2 Related articles 5 Ronin Gwenpool Mercs
for Money X-Force Deathstroke X-Men in other media Films X-Men X2 X-Men : The Last Stand soundtrack X-Men Origins : Wolverine X-Men : First Class The Wolverine
X-Men : Days of Future Past Deadpool soundtrack X-Men : Apocalypse Logan Deadpool 2 X-Men : Dark Phoenix The New Mutants Development of Gambit Characters
</item 8>

<item 9>
Deadpool 2
By April 26 , opening weekend industry projections had reached upwards of $150 million . Sequel By November 2016 , with development underway on Deadpool 2 , Fox
was also planning Deadpool 3 , which was said to include the team X-Force . With the confirmation that Leitch would direct Deadpool 2 , it was revealed that Fox
was looking for a separate filmmaker to direct Deadpool 3 . In March 2017 , Reese clarified that though Deadpool 2 sets - up the X-Force team , a future film
focused on the team would be separate from Deadpool 3 , `` so I think we 'll be able to take two paths . ( X-Force ) is where we 're launching something bigger
, but then ( Deadpool 3 is ) where we 're contracting and staying personal and small . '' After the proposed acquisition of 21st Century Fox by Disney was
announced in December 2017 , Disney CEO Bob Iger said that Deadpool would be integrated with the Marvel Cinematic Universe under Disney , and that the company
would be willing to make future R - rated Deadpool films `` as long as we let the audiences know what 's coming '' .
</item 9>

<item 10>
Deadpool 2
During filming , stunt woman Joi `` SJ '' Harris died in a motorcycle accident . Deadpool 2 is scheduled to be released in the United States on May 18 , 2018 .
A sequel , Deadpool 3 , is in development . Contents ( hide ) 1 Premise 2 Cast 3 Production 3.1 Development 3.2 Pre-production 3.3 Filming 3.4 Post-production 4
Music 5 Release 5.1 Marketing 6 Box office projection 7 Sequel 8 References 9 External links Premise Deadpool forms a team of mutants called the X-Force to
protect young mutant Russell from the time - traveling soldier Cable . Cast Ryan Reynolds as Wade Wilson / Deadpool : A wisecracking mercenary with accelerated
healing but severe scarring over his body after undergoing an experimental regenerative mutation . He forms the X-Force , a team of mutants . Josh Brolin as
Nathan Summers / Cable : A time travelling cybernetic mutant soldier , `` in many ways the opposite of Deadpool '' .
</item 10>
=== END ITEM id=c022ea0374f2e01b ===

=== ITEM id=e7421c1c2d218ea1 ===
Gold sentences and facts:
Sentence 0: Fortnite is set in contemporary Earth , where the sudden appearance of a worldwide storm causes 98 % of the world 's population to disappear , and zombie - like creatures rise to attack the remainder . Considered by Epic as a cross between Minecraft and Left 4 Dead , Fortnite has up to four players cooperating on various missions on randomly - generated maps to collect resources , build fortifications around defensive objectives that are meant to help fight the storm and protect survivors , and construct weapons and traps to engage in combat with waves of these creatures that attempt to destroy the objectives . Players gain rewards through these missions to improve their hero characters , support teams , and arsenal of weapon and trap schematics to be able to take on more difficult missions . The game is supported through microtransactions to purchase in - game currency that can be used towards these upgrades .
  Fact 0.0: Fortnite is set in contemporary Earth.
  Fact 0.1: In Fortnite, a sudden appearance of a worldwide storm causes 98% of the world's population to disappear.
  Fact 0.2: In Fortnite, zombie-like creatures rise to attack the remainder of the population.
  Fact 0.3: Epic considers Fortnite a cross between Minecraft and Left 4 Dead.
  Fact 0.4: Fortnite has up to four players cooperating on various missions.
  Fact 0.5: Fortnite missions take place on randomly-generated maps.
  Fact 0.6: Players collect resources in Fortnite missions.
  Fact 0.7: Players build fortifications around defensive objectives.
  Fact 0.8: The fortifications are meant to help fight the storm and protect survivors.
  Fact 0.9: Players construct weapons and traps.
  Fact 0.10: Players engage in combat with waves of the creatures.
  Fact 0.11: The creatures attempt to destroy the objectives.
  Fact 0.12: Players gain rewards through these missions.
  Fact 0.13: Rewards let players improve their hero characters, support teams, and arsenal of weapon and trap schematics.
  Fact 0.14: Upgrades let players take on more difficult missions.
  Fact 0.15: Fortnite is supported through microtransactions.
  Fact 0.16: Microtransactions let players purchase in-game currency.
  Fact 0.17: The in-game currency can be used towards these upgrades.
Sentence 1: Fortnite was first revealed at the 2011 Spike Video Game Awards ( VGA ) , with Epic 's former design director Cliff Bleszinski introducing a trailer for the game . Donald Mustard , creative lead at Epic , said in 2017 that this announcement was `` three weeks after we came up with the idea , before we even made the game '' . The title , which started out as an internal game jam project , represents a departure from the company 's previous work . As Bleszinski explained during the Spike event , Epic wanted to `` switch things up a little bit and do something different and fun '' with Fortnite , describing it as `` a world where you explore , you scavenge , you build and ultimately you survive . '' In an interview with Engadget , he also echoed these statements , claiming that the game would be different from the Gears of War series : `` There 's no dudebros in it ... Not that there 's anything wrong with that , right ? But creatively for the team , Gears has been amazing for us . But it 's fun to kind of stretch our wings and do something that 's a little different from the usual . '' Epic Games Vice President of Publishing Mike Fischer said in 2015 that Epic recognized that they `` announced this game too soon '' , and that its lengthy development period was due to `` very good reasons . ''
  Fact 1.0: Fortnite was first revealed at the 2011 Spike Video Game Awards (VGA).
  Fact 1.1: Epic's former design director Cliff Bleszinski introduced a trailer for Fortnite at the reveal.
  Fact 1.2: In 2017, Donald Mustard, creative lead at Epic, said the announcement came three weeks after the idea for the game was conceived.
  Fact 1.3: Donald Mustard said the announcement came before the game was made.
  Fact 1.4: Fortnite started out as an internal game jam project.
  Fact 1.5: Fortnite represents a departure from Epic's previous work.
  Fact 1.6: Bleszinski explained during the Spike event that Epic wanted to switch things up and do something different and fun with Fortnite.
  Fact 1.7: Bleszinski described Fortnite as a world where you explore, scavenge, build, and ultimately survive.
  Fact 1.8: In an interview with Engadget, Bleszinski echoed these statements.
  Fact 1.9: Bleszinski claimed Fortnite would be different from the Gears of War series.
  Fact 1.10: Bleszinski said there are no dudebros in Fortnite.
  Fact 1.11: Bleszinski said Gears of War has been amazing for the team creatively.
  Fact 1.12: Bleszinski said it is fun to stretch the team's wings and do something a little different from the usual.
  Fact 1.13: In 2015, Epic Games Vice President of Publishing Mike Fischer said Epic recognized that it announced Fortnite too soon.
  Fact 1.14: Mike Fischer said the lengthy development period of Fortnite was due to very good reasons.

Retrieved items:
<item 1>
Fortnite
Fortnite - wikipedia Fortnite Jump to : navigation , search For the battle royale version of the game , see Fortnite Battle Royale . Fortnite European box art
Developer ( s ) Epic Games People Can Fly Publisher ( s ) Epic Games Director ( s ) Darren Sugg Artist ( s ) Pete Ellis Composer ( s ) Rom Di Prisco Engine
Unreal Engine 4 Platform ( s ) Microsoft Windows macOS PlayStation 4 Xbox One iOS Release 2018 Genre ( s ) Survival Mode ( s ) Single - player , multiplayer
Fortnite is a co-op sandbox survival game developed by Epic Games and People Can Fly and published by Epic Games . The game was released as a paid - for early
access title for Microsoft Windows , macOS , PlayStation 4 and Xbox One on July 25 , 2017 , with a full free - to - play release expected in 2018 . The retail
versions of the game were published by Gearbox Publishing , while online distribution of the PC versions is handled by Epic 's launcher . Fortnite is set in
contemporary Earth , where the sudden appearance of a worldwide storm causes 98 % of the world 's population to disappear , and zombie - like creatures rise to
attack the remainder .
</item 1>

<item 2>
Fortnite
To help with this transition , the company brought in Tencent who had several games operating under this model . As part of this agreement , Tencent bought a
significant share of ownership in Epic Games through stock acquisition around June 2012 . The shifting of Epic 's approach through Tencent led to the departure
of some high - level staff later that year , including Bleszinski . Fortnite was seen as the spearhead for Epic 's games - as - a-service model as they
collaborated with Tencent . This caused some road bumps in the development of Fortnite , according to Mustard . Fortnite 's development was also slowed as it
was used as the testing ground for the new features of the Unreal 4 engine . By November 2013 , Epic confirmed that Fortnite would not release that year , nor
offered a target released date , though affirmed the game was still in development by several of its studios . Fortnite was a feature in the May 2014 issue of
Game Informer , revealing that the title would be released as a free - to - play game . By 2014 , Fortnite was at a `` pretty functional prototype '' with most
of the Unreal 4 engine elements smoothed out , according to Mustard .
</item 2>

<item 3>
Fortnite
Epic Games Vice President of Publishing Mike Fischer said in 2015 that Epic recognized that they `` announced this game too soon '' , and that its lengthy
development period was due to `` very good reasons . '' As the game was at its very preliminary stages at the VGA reveal , the goal of this reveal was to seek
public interest in the title and potential publishing partners as to decide on the game 's release platforms and timeframe . During the July 2012 San Diego
Comic Con , Epic announced that Fortnite would be an exclusive personal computer title , and the first one to be developed by Epic using their new Unreal 4 game
engine , with a planned release in 2013 . The game 's development was originally started in the Unreal 3 engine , but as they progressed , they had seen the
opportunity to work in several of the new feature sets and scripting language offered by Unreal 4 for Fortnite , while still running on most personal computers
at that time .
</item 3>

<item 4>
Fortnite
Jump up ^ Brown , Fraser ( December 10 , 2014 ) . `` Epic pulls back the curtain ( and flesh ) on Fortnite 's monster design '' . PCGamesN . Archived from the
original on July 30 , 2017 . Retrieved July 29 , 2017 . Jump up ^ Greening , Chris ( October 22 , 2016 ) . `` Game Releases Calendar : 2016 Edition '' . Game
Music Online . Archived from the original on September 30 , 2017 . Retrieved July 16 , 2017 . Jump up ^ McWhertor , Michael ( July 12 , 2012 ) . `` Epic Games '
' Fortnite ' will be the developer 's first Unreal Engine 4 game '' . Polygon . Archived from the original on July 30 , 2017 . Retrieved August 1 , 2017 . Jump
up ^ Makuch , Eddie ( July 12 , 2013 ) . `` Epic : Fortnite is `` Minecraft meets Left 4 Dead '' `` . GameSpot . Archived from the original on March 27 , 2015 .
</item 4>

<item 5>
Fortnite
`` Epic Games ' New Franchise FORTnITE Blends Survival Horror With Tower Defense Strategy '' . Forbes . Archived from the original on September 30 , 2017 .
Retrieved July 24 , 2017 . Jump up ^ Markovech , Sam ( June 8 , 2017 ) . `` Fortnite 's years of delays end with not - free - to - play version coming in July
'' . Ars Technica . Archived from the original on June 8 , 2017 . Retrieved June 8 , 2017 . ^ Jump up to : McWhertor , Michael ( March 26 , 2014 ) . `` What 's
the future of games at Epic Games ? '' . Polygon . Archived from the original on June 27 , 2017 . Retrieved June 9 , 2017 . Jump up ^ Schramm , Mike ( December
11 , 2011 ) . `` Cliff B talks Fortnite : ' There 's no dudebros in it ' '' . Engadget . Archived from the original on October 1 , 2017 .
</item 5>

<item 6>
Fortnite
Engadget . Archived from the original on October 1 , 2017 . Retrieved July 24 , 2017 . Jump up ^ Orland , Kyle ( June 8 , 2015 ) . `` Hands - on : Fortnite is
an overwhelming zombie defense experience '' . Ars Technica . Archived from the original on December 8 , 2017 . Retrieved December 7 , 2017 . Jump up ^ Yin -
Pool , Wesley ( December 12 , 2011 ) . `` Epic reveals Minecraft inspiration for Fortnite '' . Eurogamer . Archived from the original on December 1 , 2017 .
Retrieved November 29 , 2017 . Jump up ^ Rigney , Ryan ( July 12 , 2012 ) . `` First Screens : Epic 's Fortnite Is the First Unreal Engine 4 Game '' . Wired .
Archived from the original on December 22 , 2016 . Retrieved November 29 , 2017 . Jump up ^ Hafer , T.J. ( July 12 , 2012 ) .
</item 6>

<item 7>
Fortnite
They further opted for personal computer exclusivity to avoid the difficulty of having to go through console certification , and as they planned to be
constantly monitoring and tweaking the game , acting as a dungeon master , the personal computer approach would allow them to do this without restrictions
normally set by console manfuctures . Bleszinski later clarified that they would not rule out release on other platforms as they developed the title . Fortnite
cosplayers at Gamescom 2017 Fortnite 's development was spread among several of Epic 's satellite studios , and was also co-developed by the Polish studio
People Can Fly , which had worked with Epic previously on earlier games , and had been fully acquired by Epic sometime in 2012 . People Can Fly were briefly
renamed Epic Games Poland in 2013 as to align with Epic 's other studios . By March 2014 , there were about 90 developers working on the game . People Can Fly
later returned to being an independent studio and their own name in 2015 , but continued to help Epic with Fortnite 's development . During Fortnite 's
development , Epic had seen the game industry shifting to games as a service model .
</item 7>

<item 8>
Fortnite
Development History Fortnite was first revealed at the 2011 Spike Video Game Awards ( VGA ) , with Epic 's former design director Cliff Bleszinski introducing a
trailer for the game . Donald Mustard , creative lead at Epic , said in 2017 that this announcement was `` three weeks after we came up with the idea , before
we even made the game '' . The title , which started out as an internal game jam project , represents a departure from the company 's previous work . As
Bleszinski explained during the Spike event , Epic wanted to `` switch things up a little bit and do something different and fun '' with Fortnite , describing
it as `` a world where you explore , you scavenge , you build and ultimately you survive . '' In an interview with Engadget , he also echoed these statements ,
claiming that the game would be different from the Gears of War series : `` There 's no dudebros in it ... Not that there 's anything wrong with that , right ?
But creatively for the team , Gears has been amazing for us . But it 's fun to kind of stretch our wings and do something that 's a little different from the
usual . ''
</item 8>

<item 9>
Fortnite
( July 12 , 2012 ) . `` Fortnite will be a PC exclusive and Epic 's first Unreal Engine 4 game '' . PC Gamer . Archived from the original on December 1 , 2017 .
Retrieved November 29 , 2017 . Jump up ^ McGee , Maxwell ( July 13 , 2012 ) . `` Epic 's Cliff Bleszinski and Tanya Jessen Talk Up Fortnite '' . GameSpot .
Archived from the original on February 6 , 2018 . Retrieved November 29 , 2017 . ^ Jump up to : Purchase , Robert ( September 1 , 2012 ) . `` Fortnite detailed
at PAX Prime ; Minecraft meets DayZ '' . Eurogamer . Archived from the original on December 1 , 2017 . Retrieved November 29 , 2017 . Jump up ^ Sheridan ,
Connor ( November 1 , 2013 ) . `` People Can Fly working on Fortnite , renamed Epic Games Poland '' . GamesRadar . Archived from the original on December 2 ,
2017 .
</item 9>

<item 10>
Fortnite
Fortnite was being developed alongside Paragon , which Epic announced on November 2015 . As Paragon seemed to take Epic 's focus , leaving little news about
Fortnite , CEO Tim Sweeney said in March 2016 that they were still committed to Fortnite once Paragon was launched and established , given that much of the work
on Fortnite would take time to get the right balance for gameplay . `` We figure we should start with one major successful launch and do one at a time .
Fortnite will be next . '' By June 2017 , Epic Games announced that Fortnite was now set for a 2018 release across Windows , macOS , and the PlayStation 4 and
Xbox One consoles . Leading up to this free - to - play release , the game was offered as a paid early access period starting on July 25 , 2017 for all
platforms ; players who pre-ordered Founder 's Packs were granted access to the game on July 21 . The lengthy period since the game 's alpha phases was ascribed
to developing Fortnite as a games - as - a-service model , according to creative lead Donald Mustard .
</item 10>
=== END ITEM id=e7421c1c2d218ea1 ===

=== ITEM id=ff49693b3d157d67 ===
Gold sentences and facts:
Sentence 0: The Ming dynasty also engaged in a thriving trade with both Europe and Japan . The amount of silver flowing into the Ming dynasty was estimated by Joseph Needham at 300 million taels , which is equivalent to more than 190 billion dollars in today 's money . In addition to silver , the Ming also imported many European firearms , in order to ensure the modernness of their weapons .
  Fact 0.0: The Ming dynasty engaged in a thriving trade with Europe.
  Fact 0.1: The Ming dynasty engaged in a thriving trade with Japan.
  Fact 0.2: Joseph Needham estimated the amount of silver flowing into the Ming dynasty at 300 million taels.
  Fact 0.3: The estimated 300 million taels of silver is equivalent to more than 190 billion dollars in today's money.
  Fact 0.4: The Ming imported many European firearms in addition to silver.
  Fact 0.5: The Ming imported European firearms to ensure the modernness of their weapons.

Retrieved items:
<item 1>
Economy of the Ming dynasty
Many of these markets appeared in the rural countryside , where goods were exchanged and bartered . A second type of market that developed in China was the
urban - rural type , in which rural goods were sold to urban dwellers . This was particular the case when landlords decided to reside in the cities , and use
income coming from rural land holding to facilitate exchange in the cities . Another way this type of market was used was professional merchants who bought
rural goods in large quantities . The third type of market was the `` national market '' which was developed during the Song dynasty but particularly enhanced
during the Ming . This market involved not only the exchange described above , but also products produced directly for the market . Unlike earlier dynasties ,
many Ming peasants were no longer producing only products they needed ; many of them produced products for the market , which then they sold at a profit . Trade
and investment ( edit ) Main article : Haijin In the early Ming , after the devastation of the war which expelled the Mongols , the Hongwu Emperor imposed
severe restrictions on trade ( the `` haijin '' ) .
</item 1>

<item 2>
Economy of the Ming dynasty
These privately made `` sycee '' first came into use in Guangdong , spreading to the lower Yangtze sometime before 1423 , the year it became acceptable for
payment of tax obligations . In the mid-15th century , the paucity of circulating silver caused a monetary contraction and extensive reversion to barter . The
problem was met through smuggled , then legal , importation of Japanese silver ( mostly through the Portuguese and Dutch ) and Spanish silver from Potosí
carried on the Manila galleons . Provincial taxes were required to be paid in silver in 1465 ; the salt tax , in 1475 ; and corvée exemptions , in 1485 . By the
late Ming , the amount of silver being used was extraordinary : at a time when English traders considered tens of thousands of pounds an exceptional fortune ,
the Zheng clan of merchants regularly engaged in transactions valued at millions of taels .
</item 2>

<item 3>
Economy of the Ming dynasty
( eds . ) , The Ming Dynasty , 1 398 -- 1644 , Part 2 , The Cambridge History of China , 8 , Cambridge : Cambridge University Press , pp. 106 -- 172 , ISBN 978
- 0 - 521 - 24333 - 9 CS1 maint : Uses editors parameter ( link ) External links ( edit ) China 's GDP in the Ming Dynasty Guan Hanhui David Daokui Li Ming
dynasty topics History Early ( 1368 -- 1435 ) Red Turban Rebellion Ming conquest of Yunnan Ming - Mong Mao War Ming campaign against the Uriankhai Battle of
Buir Lake Ming -- Mong Mao Intervention Jingnan Campaign Ming treasure voyages Ming -- Turpan conflict Ming -- Hồ War Battle of Kherlen Ming -- Kotte War Lam
Sơn uprising Yongle Emperor 's campaigns against the Mongols Rule of Ren and Xuan Middle ( 1435 -- 1572 ) Luchuan -- Pingmian campaigns Tumu Crisis Rebellion of
Cao Qin Miao Rebellions Prince of Anhua rebellion Prince of Ning rebellion Malayan -- Portuguese war Japanese missions to Ming China Ningbo Incident Great Rites
Controversy
</item 3>

<item 4>
Economy of the Ming dynasty
A new class of wage laborers sprung up where none had existed before . In Jingde alone , it was reported that there were no less than 300 pottery factories ,
all operated by wage laborers . Early encouragement of agriculture under Hongwu ( edit ) In order to recover from rule of the Mongols and the wars that followed
them , the Hongwu Emperor enacted pro-agricultural policies . The state invested extensively in agricultural canals , reduced taxes on agriculture to 1 / 30 of
the output , and later to 1.5 % of agricultural output . Ming farmers also introduced many innovations such as water - powered plows , and new agricultural
methods such as crop rotation . This led to a massive agricultural surplus that became the basis of a market economy . Emergence of commercial plantations (
edit ) The Ming saw the rise of commercial plantations who produced crops suitable to their regions . Tea , fruits , paint and other goods were produced on a
massive scale by these agricultural plantations . Regional patterns of production established during this period continued into the Qing dynasty . The Columbian
exchange brought crops such as corn with these foreign crops .
</item 4>

<item 5>
Economy of the Ming dynasty
The entire foreign trade , which was estimated at up to 300 million taels , provided the Ming with a tax of only about 40,000 taels a year . When the Wanli
Emperor sought to increase the salt tax , his measures were opposed by violence and the eunuchs he sent to collect the tax were beheaded by local officials .
Sprouts of capitalism ( edit ) Investment and capital moved off the land and were poured into ventures . Continuing the trend from the Song , Ming investors
poured large amounts of capital into ventures and reaped high profits . Many Chinese scholars believe the Ming was the dynasty in which the `` sprouts of
capitalism '' emerged in China , only to be suppressed by the Qing Dynastry . This theory was widely promoted by Communist scholars during the Maoist period ,
and suffers from that period 's general condemnation of the Manchu Qing , who were accused of mis - managing the Chinese state in the face of foreign
encroachment .
</item 5>

<item 6>
Economy of the Ming dynasty
Believing that agriculture was the basis of the economy , Hongwu favored that industry over all else , including that of merchants . After his death , most of
his policies were reversed by his successors . By the late Ming , the state was losing power to the very merchants which Hongwu had wanted to restrict . The
Ming dynasty also engaged in a thriving trade with both Europe and Japan . The amount of silver flowing into the Ming dynasty was estimated by Joseph Needham at
300 million taels , which is equivalent to more than 190 billion dollars in today 's money . In addition to silver , the Ming also imported many European
firearms , in order to ensure the modernness of their weapons . Trade and commerce thrived in this liberalized economy , and was aided by the construction of
canals , roads , and bridges by the Ming government . The Ming saw the rise of several merchant clans such as the Huai and Jin clans , who disposed of large
amounts of wealth . The gentry and merchant classes started to fuse , and the merchants gained power at the expense of the state . Some merchants were reputed
to have a treasure of 30 million taels .
</item 6>

<item 7>
Economy of the Ming dynasty
However , a second silver contraction occurred in the mid-17th century when King Philip IV began enforcing laws limiting direct trade between Spanish South
America and China at about the same time the new Tokugawa shogunate in Japan restricted most of its foreign exports , cutting off Dutch and Portuguese access to
its silver . The dramatic spike in silver 's value in China made payment of taxes nearly impossible for most provinces . In extremis , the government even
resumed use of paper currency amid Li Zicheng 's rebellion . Manufactures ( edit ) Privatization ( edit ) Another key feature of the Ming manufacturing industry
was privatization . Unlike the Song , in which state - owned enterprises played a large role , the Ming reverted to the old laissez faire policies of the Han by
privatizing the salt and tea industries . By the middle of the Ming dynasty , powerful groups of wealthy merchants had replaced the state as the dominant movers
behind Chinese industry . Emergence of wage labor ( edit ) The Ming government abolished the mandatory forced labor by peasants used in early dynasties and
replaced it with wage labor . A new class of wage laborers sprung up where none had existed before .
</item 7>

<item 8>
Economy of the Ming dynasty
Contents ( hide ) 1 Currency 2 Manufactures 2.1 Privatization 2.2 Emergence of wage labor 3 Early encouragement of agriculture under Hongwu 3.1 Emergence of
commercial plantations 3.2 Rural markets during the Ming 4 Trade and investment 4.1 Taxation 4.2 Weakening of the state 4.3 Sprouts of capitalism 5 See also 6
References 7 External links Currency ( edit ) Further information : Ming dynasty coinage The early Ming dynasty attempted to use paper currency , with outflows
of bullion limited by its ban on private foreign commerce . Like its forebears , the currency experienced massive counterfeiting and hyperinflation . ( In 1425
, Ming notes were trading at about 0.014 % of its original value under the Hongwu Emperor . ) The notes remained in circulation as late as 1573 but their
printing was ended in 1450 . Minor coins were minted in base metals , but trade mostly occurred using silver ingots . As their purity and exact weight varied ,
they were treated as bullion and measured in tael .
</item 8>

<item 9>
Economy of the Ming dynasty
The Columbian exchange brought crops such as corn with these foreign crops . During the Ming , specialized areas also popped up planting large numbers of cash
crops that could be sold at markets . Large numbers of peasants abandoned the land to become artisans . The population of the Ming boomed ; estimates for the
population of the Ming range from 160 to 200 million . Rural markets during the Ming ( edit ) Ming agriculture was much changed from the earlier areas ; firstly
, gigantic areas , devoting and specializing in cash crops , sprung up to demand from the new market economy . Secondly , agricultural tools and carts , some
water - powered , help to create a gigantic agricultural surplus which formed the basis of the rural economy . Besides rice , other crops were grown on a large
scale . Although images of autarkic farmers who had no connection to the rest of China may have some merit for the earlier Han and T'ang dynasties , this was
certainly not the case for the Ming dynasty . During the Ming dynasty , the increase in population and the decrease in quality land made it necessary that
farmers make a living off cash crops .
</item 9>

<item 10>
Economy of the Ming dynasty
Economy of the Ming dynasty - wikipedia Economy of the Ming dynasty Jump to : navigation , search This article needs additional citations for verification .
Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( October 2010 ) ( Learn how and
when to remove this template message ) The Ming Empire around 1580 The economy of the Ming dynasty ( 1368 -- 1644 ) of China was the largest in the world during
that period . It is regarded as one of China 's three golden ages ( the other two being the Han and Song periods ) . The period was marked by the increasing
political influence of the merchants , the gradual weakening of imperial rule , and technological advances .
</item 10>
=== END ITEM id=ff49693b3d157d67 ===

=== ITEM id=ddd9a45f97163bf6 ===
Gold sentences and facts:
Sentence 0: Shortwave radio is radio transmission using shortwave radio frequencies . There is no official definition of the band , but the range always includes all of the high frequency band ( HF ) , and generally extends from 1.7 -- 30 MHz ( 176.3 -- 10.0 m ) ; from the high end of the medium frequency band ( MF ) just above the mediumwave AM broadcast band , to the end of the HF band .
  Fact 0.0: Shortwave radio is radio transmission using shortwave radio frequencies.
  Fact 0.1: There is no official definition of the shortwave band.
  Fact 0.2: The shortwave range always includes the high frequency band (HF).
  Fact 0.3: The shortwave range generally extends from 1.7 to 30 MHz (176.3 to 10.0 m).
  Fact 0.4: The shortwave range extends from the high end of the medium frequency band (MF), just above the mediumwave AM broadcast band, to the end of the HF band.
Sentence 1: Very little infrastructure is required for long - distance two - way communications using shortwave radio . All one needs is a pair of transceivers , each with an antenna , and a source of energy ( such as a battery , a portable generator , or the electrical grid ) . This makes shortwave radio one of the most robust means of communications , which can be disrupted only by interference or bad ionospheric conditions . Modern digital transmission modes such as MFSK and Olivia are even more robust , allowing successful reception of signals well below the noise floor of a conventional receiver .
  Fact 1.0: Very little infrastructure is required for long-distance two-way communications using shortwave radio.
  Fact 1.1: Long-distance two-way shortwave communication needs a pair of transceivers, each with an antenna.
  Fact 1.2: Long-distance two-way shortwave communication needs a source of energy, such as a battery, a portable generator, or the electrical grid.
  Fact 1.3: Shortwave radio is one of the most robust means of communications.
  Fact 1.4: Shortwave communications can be disrupted only by interference or bad ionospheric conditions.
  Fact 1.5: Modern digital transmission modes such as MFSK and Olivia are even more robust than conventional shortwave modes.
  Fact 1.6: MFSK and Olivia allow successful reception of signals well below the noise floor of a conventional receiver.

Retrieved items:
<item 1>
Shortwave radio
The drawback is the receiver is more complicated , since it must re-create the carrier to recover the signal . Small errors in the detection process greatly
affect the pitch of the received signal . As a result , single sideband is not used for music or general broadcast . Single sideband is used for long - range
voice communications by ships and aircraft , Citizen 's Band , and amateur radio operators . Lower sideband ( LSB ) is generally used below 9 MHz and USB (
upper sideband ) above 9 MHz . Vestigal sideband transmits the carrier and one complete sideband , but filters out the other sideband . It is a compromise
between AM and SSB , enabing simple receivers to be used , but requires almost as much transmitter power as AM . One advantage is only half the bandwidth of an
AM signal is used . It can be heard in the transmission of certain radio time signal stations . Vestigial sideband is used for over the air Television
Broadcasts both analog and digital .
</item 1>

<item 2>
Shortwave radio
See also ( edit ) ALLISS -- a very large rotatable antenna system used in international broadcasting Amateur radio -- also uses the shortwave bands , but with
power levels under 2 kW International broadcasting List of American shortwave broadcasters List of shortwave radio broadcasters Long wave Medium wave Shortwave
bands -- shortwave spectrum allocation shortwave relay station -- the fundamental way in which programmes are broadcast on shortwave SSB -- a method of radio
signal modulation References ( edit ) Jump up ^ Grundig Satellit 400 international / professional ShortWaveRadio.ch Jump up ^ Inconsistent article :
itwissen.info : KW ( Kurzwelle ) SW ( short wave ) Quote : `` ... Der KW - Frequenzbereich ( SW ) liegt zwischen 3 MHz und 30 MHz ( < - 1 . definition ) ...
Kurzwelle ist auch eine sendetechnische Bezeichnung für Rundfunk im Frequenzbereich zwischen 2 , 3 MHz und 26 , 1 MHz ( < - 2 .
</item 2>

<item 3>
Shortwave radio
The last WRC took place in 2007 . At WRC - 97 in 1997 , the following bands were allocated for international broadcasting . AM shortwave broadcasting channels
are allocated with a 5 kHz separation for traditional analog audio broadcasting . Metre Band Frequency Range Remarks 120 m 2.3 -- 2.495 MHz tropical band 90 m
3.2 -- 3.4 MHz tropical band 75 m 3.9 -- 4 MHz shared with the North American amateur radio 80m band 60 m 4.75 -- 5.06 MHz tropical band 49 m 5.9 -- 6.2 MHz 41
m 7.2 -- 7.6 MHz shared with the amateur radio 40m band 31 m 9.4 -- 9.9 MHz currently the most heavily used band 25 m 11.6 -- 12.2 MHz 22 m 13.57 -- 13.87 MHz
19 m 15.1 -- 15.8 MHz 16 m 17.48 -- 17.9 MHz 15 m 18.9 -- 19.02 MHz almost unused ,
</item 3>

<item 4>
Shortwave radio
Continuous wave ( CW ) is on - and - off keying of a carrier , used for Morse code communications and Hellschreiber facsimile - based teleprinter transmissions
. Narrow - band frequency modulation ( NBFM ) is used typically above 20 MHz . Because of the larger bandwidth required , NBFM is commonly used for VHF
communication . Regulations limit the bandwidth of a signal transmitted in the HF bands , and the advantages of frequency modulation are greatest if the FM
signal has a wide bandwidth . NBFM is limited to short - range transmissions due to the multiphasic distortions created by the ionosphere . Digital Radio
Mondiale ( DRM ) is a digital modulation for use on bands below 30 MHz . Radioteletype , fax , digital , slow - scan television and other systems use forms of
frequency - shift keying or audio subcarriers on a shortwave carrier . These generally require special equipment to decode , such as software on a computer
equipped with a sound card .
</item 4>

<item 5>
Shortwave radio
9 MHz 15 m 18.9 -- 19.02 MHz almost unused , could become a DRM band 13 m 21.45 -- 21.85 MHz 11 m 25.6 -- 26.1 MHz may be used for local DRM broadcasting
Although countries generally follow the table above , there may be small differences between countries or regions . For example , in the official bandplan of
the Netherlands , the 49 m band starts at 5.95 MHz , the 41 m band ends at 7.45 MHz , the 11 m band starts at 25.67 MHz , and the 120 , 90 and 60 m bands are
absent altogether . Additionally , international broadcasters sometimes operate outside the normal WRC - allocated bands or use off - channel frequencies . This
is done for practical reasons , or to attract attention in crowded bands ( 60m , 49m , 40m , 41m , 31m , 25m ) . The new digital audio broadcasting format for
shortwave DRM operates 10 kHz or 20 kHz channels .
</item 5>

<item 6>
Shortwave radio
Contents ( hide ) 1 Frequency classifications 2 History 2.1 Development 2.2 Amateur use of shortwave propagation 3 Propagation characteristics 4 Types of
modulation 5 Users 6 Shortwave broadcasting 6.1 Frequency allocations 6.2 Advantages 6.3 Disadvantages 7 Shortwave listening 8 Shortwave broadcasts and music 9
Shortwave 's future 10 See also 11 References 12 External links Frequency classifications ( edit ) The widest popular definition of the shortwave frequency
interval is the ITU Region 1 ( EU + Africa + Russia ... ) definition , and is the span 1.6 -- 30 MHz , just above the medium wave band , which ends
approximately at 1.6 MHz .
</item 6>

<item 7>
Shortwave radio
Shortwave radio received its name because the wavelengths in this band are relatively shorter than 200 m ( 1,500 kHz ) which marked the original upper limit of
the medium frequency band first used for radio communications . The broadcast medium wave band now extends above the 200 m / 1,500 kHz limit , and the amateur
radio 1.8 MHz -- 2.0 MHz band ( known as the `` top band '' ) is the lowest - frequency band considered to be ' shortwave ' . Early long distance radio
telegraphy used long waves , below 300 kilohertz ( kHz ) . The drawbacks to this system included a very limited spectrum available for long distance
communication , and the very expensive transmitters , receivers and gigantic antennas that were required . It was also difficult to beam the radio wave
directionally with long wave , resulting in a major loss of power over long distances . Prior to the 1920s , the shortwave frequencies above 1.5 MHz were
regarded as useless for long distance communication and were designated in many countries for amateur use .
</item 7>

<item 8>
Shortwave radio
Users ( edit ) Portable shortwave receiver 's digital display tuned to the 75 meter band Some established users of the shortwave radio bands may include :
International broadcasting primarily by government - sponsored propaganda , international news ( for example , the BBC World Service ) or cultural stations to
foreign audiences : the most common use of all . Domestic broadcasting : to widely dispersed populations with few longwave , mediumwave and FM stations serving
them ; or for specialty political , religious and alternative media networks ; or of individual commercial and non-commercial paid broadcasts . Oceanic air
traffic control uses the HF / shortwave band for long distance communication to aircraft over the oceans and poles , which are far beyond the range of
traditional VHF frequencies . Modern systems also include satellite communications , such as ADS - C / CPDLC `` Utility '' stations transmitting messages not
intended for the general public , such as merchant shipping , marine weather , and ship - to - shore stations ; for aviation weather and air - to - ground
communications ; for military communications ; for long - distance governmental purposes , and for other non-broadcast communications .
</item 8>

<item 9>
Shortwave radio
L band S band Digital systems ADR DAB - S DVB - SH S - DMB SDR Commercial radio providers 1worldspace Sirius XM Holdings SiriusXM Canada Codecs AAC AMR - WB+
HDC HE - AAC MPEG - 1 Audio Layer II Subcarrier signals AMSS DirectBand PAD RDS / RBDS SCA / SCMO DARC Related topics Technical ( audio ) Audio data compression
Audio signal processing Technical ( AM stereo formats ) Belar C - QUAM Harris Kahn - Hazeltine Magnavox Technical ( emission ) AM broadcasting AM expanded band
Cable radio Digital radio Error detection and correction FM broadcast band FM broadcasting Multipath propagation Shortwave relay station Cultural History of
radio International broadcasting Comparison of radio systems Radio spectrum ( ITU ) ELF 3 Hz / 100 Mm 30 Hz / 10 Mm SLF 30 Hz / 10 Mm 300 Hz / 1 Mm ULF 300 Hz /
1 Mm 3 kHz / 100 km VLF 3 kHz / 100 km 30 kHz / 10 km LF 30 kHz / 10 km 300 kHz / 1 km MF 300 kHz / 1 km 3 MHz / 100 m HF
</item 9>

<item 10>
Shortwave radio
`` Short - Wave Radio Telephone is Success in Tests '' Popular Mechanics , July 1931 , mid page experiments carried out for the French and British governments
Que Escuchar en la Onda Corta en Español website Time signal stations Longwave BPC BPL BSF DCF77 JJY RBU RJH66 ( Beta ) RTZ MSF TDF WWVB Shortwave BPM BSF CHU
HD2IOA HLA JN53DV RWM ROA WWV WWVH ( Hawaii , USA ) YVTO VHF / FM / UHF RDS CT Satellite GPS BeiDou Galileo GLONASS IRNSS QZSS Defunct HBG NAA WWVL OMA OLB5 VNG
Y3S Analog and digital audio broadcasting Terrestrial Radio modulation AM FM COFDM Frequency allocations LW ( LF ) MW ( MF ) SW ( HF ) VHF ( low / mid / high )
L band ( UHF ) Digital systems CAM - D DAB / DAB+ DRM / DRM+ FMeXtra HD Radio CDR DVB - T2 Lite Satellite Frequency allocations C band K band L band S band
Digital systems ADR DAB - S DVB - SH S - DMB
</item 10>
=== END ITEM id=ddd9a45f97163bf6 ===
