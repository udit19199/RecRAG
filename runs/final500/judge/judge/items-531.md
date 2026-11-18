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

This file has 12 items.

=== ITEM id=c7f592b1c6c4ba3e ===
Gold sentences and facts:
Sentence 0: Prof. Dr. Ernst Willi Messerschmid (born May 21, 1945) is a German physicist and former astronaut.
  Fact 0.0: Ernst Willi Messerschmid is Prof. Dr.
  Fact 0.1: Ernst Willi Messerschmid was born on May 21, 1945.
  Fact 0.2: Ernst Willi Messerschmid is a German physicist.
  Fact 0.3: Ernst Willi Messerschmid is a former astronaut.
Sentence 1: Philippe Perrin (Colonel, French Air Force) (born January 6, 1963) is a French test pilot and former CNES and European Space Agency astronaut.
  Fact 1.0: Philippe Perrin was born on January 6, 1963.
  Fact 1.1: Philippe Perrin is a Colonel in the French Air Force.
  Fact 1.2: Philippe Perrin is a French test pilot.
  Fact 1.3: Philippe Perrin is a former CNES astronaut.
  Fact 1.4: Philippe Perrin is a former European Space Agency astronaut.

Retrieved items:
<item 1>
Ernst Messerschmid
Prof. Dr. Ernst Willi Messerschmid (born May 21, 1945) is a German physicist and former astronaut.
</item 1>

<item 2>
Serge Haroche
Serge Haroche (born 11 September 1944) is a French physicist who was awarded the 2012 Nobel Prize for Physics jointly with David J. Wineland for
"ground-breaking experimental methods that enable measuring and manipulation of individual quantum systems", a study of the particle of light, the photon.

 This and his other works developed laser spectroscopy.

 Since 2001, Haroche is a Professor at the Collège de France and holds the Chair of Quantum Physics.

 In 1971 he defended his doctoral thesis in physics at the University of Paris VI, his research has been conducted under the direction of Claude
Cohen-Tannoudji.
</item 2>

<item 3>
François Englert
François Baron Englert (] ; born 6 November 1932) is a Belgian theoretical physicist and 2013 Nobel prize laureate (shared with Peter Higgs).

 He is Professor emeritus at the Université libre de Bruxelles (ULB) where he is member of the Service de Physique Théorique.

 He is also a Sackler Professor by Special Appointment in the School of Physics and Astronomy at Tel Aviv University and a member of the Institute for Quantum
Studies at Chapman University in California.

 He was awarded the 2010 J. J. Sakurai Prize for Theoretical Particle Physics (with Gerry Guralnik, C. R. Hagen, Tom Kibble, Peter Higgs, and Robert Brout), the
Wolf Prize in Physics in 2004 (with Brout and Higgs) and the High Energy and Particle Prize of the European Physical Society (with Brout and Higgs) in 1997 for
the mechanism which unifies short and long range interactions by generating massive gauge vector bosons.

 He has made contributions in statistical physics, quantum field theory, cosmology, string theory and supergravity.
</item 3>

<item 4>
Hermann Knoblauch
Karl Hermann Knoblauch (11 April 1820 – 30 June 1895) was a German physicist.

 He is most notable for his studies of radiant heat.

 He was one of the six founding members of the Deutsche Physikalische Gesellschaft at Berlin on 14 January 1845.
</item 4>

<item 5>
Philippe Perrin
Philippe Perrin (Colonel, French Air Force) (born January 6, 1963) is a French test pilot and former CNES and European Space Agency astronaut.
</item 5>

<item 6>
Émile Verdet
Marcel Émile Verdet (13 March 1824 – 3 June 1866) was a French physicist.

 He worked in magnetism and optics, editing the works of Augustin-Jean Fresnel.

 Verdet did much to champion the early theory of the conservation of energy in France through his editorial supervision of the "Annales de chimie et de
physique".
</item 6>

<item 7>
Alfred von Arneth
Alfred Ritter von Arneth (10 July 181930 July 1897) was an Austrian historian.

 His principal scholarly work is a ten-volume biography of the Habsburg Empress Maria Theresa, first published in installments from 1863 to 1879 and still
regarded as the standard work on the subject.

 Born at Vienna, he was the son of Joseph Calasanza von Arneth (1791–1863), a well-known historian and archaeologist, who wrote a history of the Austrian Empire
(Vienna, 1827) and several works on numismatics and brother of Doctor Franz Hektor von Arneth (1818–1907).
</item 7>

<item 8>
Philippe Perrin (artist)
Philippe Perrin, (La Tronche, 10 August 1964) is a French artist and contemporary photographer who lives and works in Paris.
</item 8>

<item 9>
François Englert
He is the recipient of the 2013 Prince of Asturias Award in technical and scientific research, together with Peter Higgs and the CERN.
</item 9>

<item 10>
Hans Schlegel
Hans Wilhelm Schlegel (Überlingen, 3 August 1951) is a German physicist, an ESA astronaut, and a veteran of two NASA Space Shuttle missions.
</item 10>
=== END ITEM id=c7f592b1c6c4ba3e ===

=== ITEM id=f244ddaf3faa7eac ===
Gold sentences and facts:
Sentence 0: The Karlsschrein (English: Shrine of Charlemagne ) in Aachen Cathedral was made in Aachen at the command of Frederick II, Holy Roman Emperor and completed in 1215, after Frederick II's grandfather, Frederick Barbarossa had exhumed Charlemagne's bones from their resting place in the Palatine Chapel, Aachen in 1165.
  Fact 0.0: The Karlsschrein, English name Shrine of Charlemagne, is in Aachen Cathedral.
  Fact 0.1: The Karlsschrein was made in Aachen.
  Fact 0.2: The Karlsschrein was made at the command of Frederick II, Holy Roman Emperor.
  Fact 0.3: The Karlsschrein was completed in 1215.
  Fact 0.4: Frederick II's grandfather was Frederick Barbarossa.
  Fact 0.5: Frederick Barbarossa exhumed Charlemagne's bones from their resting place in the Palatine Chapel, Aachen.
  Fact 0.6: Frederick Barbarossa exhumed Charlemagne's bones in 1165.
  Fact 0.7: The exhumation of Charlemagne's bones happened before the Karlsschrein was completed.
Sentence 1: Frederick I (German: "Friedrich I" , Italian: "Federico I" ; 1122 – 10 June 1190), also known as Frederick Barbarossa (Italian: "Federico Barbarossa" ), was the Holy Roman Emperor from 1155 until his death.
  Fact 1.0: Frederick I's German name is "Friedrich I".
  Fact 1.1: Frederick I's Italian name is "Federico I".
  Fact 1.2: Frederick I lived from 1122 to 10 June 1190.
  Fact 1.3: Frederick I is also known as Frederick Barbarossa.
  Fact 1.4: Frederick Barbarossa's Italian name is "Federico Barbarossa".
  Fact 1.5: Frederick I was the Holy Roman Emperor from 1155 until his death.

Retrieved items:
<item 1>
Eric II of Denmark
Eric II the Memorable (Danish: "Erik II Emune" ) (  1090 – 18 September 1137) was king of Denmark between 1134 and 1137.

 Eric was an illegitimate son of Eric I of Denmark, who ruled Denmark from 1095 to 1103.

 Eric the Memorable rebelled against his uncle Niels of Denmark, and was declared king in 1134.

 He punished his adversaries severely, and rewarded his supporters handsomely.

 He was killed by a subject in 1137 and was promptly succeeded by his nephew Eric III of Denmark.
</item 1>

<item 2>
Eric III of Denmark
Eric III Lamb (Danish: "Erik III Lam" ) (  1120 – 27 August 1146) was the King of Denmark from 1137 until 1146.

 He was the grandson of Eric I of Denmark and the nephew of Eric II of Denmark, whom he succeeded on the throne.

 He abdicated in 1146, as the first and only Danish monarch to do so.

 His succession led to a period of civil war between Sweyn III, Canute V, and Valdemar I.
</item 2>

<item 3>
Frederick I, Holy Roman Emperor
Frederick I (German: "Friedrich I" , Italian: "Federico I" ; 1122 – 10 June 1190), also known as Frederick Barbarossa (Italian: "Federico Barbarossa" ), was the
Holy Roman Emperor from 1155 until his death.

 He was elected King of Germany at Frankfurt on 4 March 1152 and crowned in Aachen on 9 March 1152.

 He became King of Italy in 1155 and was crowned Roman Emperor by Pope Adrian IV on 18 June 1155.

 Two years later, the term "sacrum " ("holy") first appeared in a document in connection with his Empire.

 He was later formally crowned King of Burgundy, at Arles on 30 June 1178.

 He was named "Barbarossa " by the northern Italian cities which he attempted to rule: Barbarossa means "red beard" in Italian; in German, he was known as
"Kaiser Rotbart ", which has the same meaning.
</item 3>

<item 4>
List of Holy Roman Empresses
Holy Roman Empress or Empress of the Holy Roman Empire is the title given to the consort (wife) or regent of the Holy Roman Emperor.

 The elective dignity of Holy Roman Emperor was restricted to males only, therefore there was never a Holy Roman Empress regnant, though women such as Theophanu
or Maria Theresa of Austria, who controlled the power of rule, served as de facto Empresses regnant.
</item 4>

<item 5>
Harald II of Denmark
Harald II of Denmark (died 1018) was King of Denmark from 1014 until his death in 1018.

 He was the youngest son of Sweyn Forkbeard and Gunhild of Wenden, and was regent while his father was fighting Ethelred the Unready in England.

 He inherited the Danish throne in 1014, and held it while his brother, the later king Cnut the Great conquered England.

 After his death in 1018, he was succeeded by Cnut the Great.

 Little detail is known about Harald II.
</item 5>

<item 6>
Prince Christian of Hanover (born 1985)
Prince Christian of Hanover (Christian Heinrich Clemens Paul Frank Peter Welf Wilhelm-Ernst Friedrich Franz Prince of Hanover, Duke of Brunswick and Lüneburg;
born 1 June 1985) is the younger son of Ernst August, Prince of Hanover, and his first wife Chantal Hochuli.

 He is the second in the line of succession to the former Hanoverian throne, after his elder brother Prince Ernst August.

 As a descendant of George III of the United Kingdom, Christian is also in the line of succession to the British throne.
</item 6>

<item 7>
Karlsschrein
The Karlsschrein (English: Shrine of Charlemagne ) in Aachen Cathedral was made in Aachen at the command of Frederick II, Holy Roman Emperor and completed in
1215, after Frederick II's grandfather, Frederick Barbarossa had exhumed Charlemagne's bones from their resting place in the Palatine Chapel, Aachen in 1165.
</item 7>

<item 8>
Karl Anselm, 4th Prince of Thurn and Taxis
Karl Anselm, 4th Prince of Thurn and Taxis, full German name: "Karl Anselm Fürst von Thurn und Taxis" (2 June 1733, Frankfurt am Main, Free Imperial City of
Frankfurt, Holy Roman Empire – 13 November 1805, Winzer bei Regensburg, Electorate of Bavaria, Holy Roman Empire) was the fourth Prince of Thurn and Taxis,
Postmaster General of the Imperial Reichspost, and Head of the Princely House of Thurn and Taxis from 17 March 1773 until his death on 13 November 1805.

 Karl Anselm served as "Prinzipalkommissar" at the Perpetual Imperial Diet in Regensburg for Joseph II, Holy Roman Emperor and Francis II, Holy Roman Emperor
from 1773 to 1797.
</item 8>

<item 9>
Gottlieb Muffat
Gottlieb Muffat (April 1690 – 9 December 1770), son of Georg Muffat, served as "Hofscholar" under Johann Fux in Vienna from 1711 and was appointed to the
position of third court organist at the "Hofkapelle" in 1717.

 He acquired additional duties over time including the instruction of members of the Imperial family, among them the future Empress Maria Theresa.

 He was promoted to second organist in 1729 and first organist upon the accession of Maria Theresa to the throne in 1741.

 He retired from official duties at the court in 1763.
</item 9>

<item 10>
Ernest Augustus, King of Hanover
Ernest Augustus (5 June 1771 – 18 November 1851) was King of Hanover from 20 June 1837 until his death.

 He was the fifth son and eighth child of King George III of the United Kingdom and Hanover.

 As a fifth son, Ernest seemed unlikely to become a monarch, but none of his four elder brothers had a legitimate son who survived infancy.

 The Salic Law, which barred succession to or through a female, prevailed in Hanover; therefore, when his elder brother King William IV died in 1837, Ernest
succeeded him as King of Hanover.

 In the United Kingdom the succession to the monarchy was determined by primogeniture and his niece Victoria succeeded to the throne, thus ending the personal
union between the British Isles and Hanover that had existed since 1714.
</item 10>
=== END ITEM id=f244ddaf3faa7eac ===

=== ITEM id=b8f4d7cd7ba762b4 ===
Gold sentences and facts:
Sentence 0: That Darn Cat is a 1997 American mystery comedy film starring Christina Ricci and Doug E. Doug.
  Fact 0.0: That Darn Cat is a film.
  Fact 0.1: That Darn Cat is from 1997.
  Fact 0.2: That Darn Cat is American.
  Fact 0.3: That Darn Cat is a mystery comedy.
  Fact 0.4: That Darn Cat stars Christina Ricci.
  Fact 0.5: That Darn Cat stars Doug E. Doug.
Sentence 1: Honey, I Blew Up the Kid is a 1992 American comedy science fiction film and the sequel to the 1989 film "Honey, I Shrunk the Kids".
  Fact 1.0: Honey, I Blew Up the Kid is a film.
  Fact 1.1: Honey, I Blew Up the Kid is from 1992.
  Fact 1.2: Honey, I Blew Up the Kid is American.
  Fact 1.3: Honey, I Blew Up the Kid is a comedy science fiction film.
  Fact 1.4: Honey, I Blew Up the Kid is a sequel to Honey, I Shrunk the Kids.
  Fact 1.5: Honey, I Shrunk the Kids is a 1989 film.

Retrieved items:
<item 1>
The Apple Dumpling Gang
The Apple Dumpling Gang is a 1971 novel by Jack Bickham, about a group of orphaned children during the California gold rush.

 They encounter a gambler who reluctantly helps them, as well as a pair of hapless robbers who are after the gold the children have found.
</item 1>

<item 2>
That Darn Cat (1997 film)
That Darn Cat is a 1997 American mystery comedy film starring Christina Ricci and Doug E. Doug.

 It is a remake of the 1965 film "That Darn Cat!

", which in turn was based on the book "Undercover Cat" by Gordon and Mildred Gordon.

 It is directed by British TV veteran Bob Spiers (most famous for "Fawlty Towers", as well as "Spice World") and written by Scott Alexander and Larry
Karaszewski, best known for "Ed Wood" and the first two "Problem Child" films.
</item 2>

<item 3>
That Darn Cat!
That Darn Cat!

 is a 1965 American Walt Disney Productions thriller comedy film starring Hayley Mills (in her last of the six films she made for the Walt Disney Studios) and
Dean Jones (starring in his first film for Disney) in a story about bank robbers, a kidnapping and a mischievous cat.

 The film was based on the 1963 novel "Undercover Cat" by Gordon and Mildred Gordon and was directed by Robert Stevenson.

 The title song was written by the Sherman Brothers and sung by Bobby Darin.

 The 1997 remake includes a cameo appearance by Dean Jones.
</item 3>

<item 4>
Codename: Kids Next Door
Codename: Kids Next Door, commonly abbreviated to Kids Next Door or KND, is an American animated television series created by Tom Warburton for Cartoon Network,
and the 13th of the network's "Cartoon Cartoons".

 The series centers on the adventures of five children who operate from a high-tech tree house, fighting against adult and teen villains with advanced 2×4
technology.

 Using their codenames (Numbuh 1, Numbuh 2, Numbuh 3, Numbuh 4, and Numbuh 5), they are Sector V, part of a global organization called the Kids Next Door.
</item 4>

<item 5>
Amy O'Neill
Amy O'Neill (born July 8, 1971) is an American performer and former actress.

 After appearing in several sitcoms and starring as Molly Stark on "The Young and the Restless" in 1986, she was cast in her notable role as Amy Szalinski in
the 1989 Disney film, "Honey, I Shrunk the Kids", for which she was nominated for a Young Artist Award.

 She reprised her role as Amy Szalinski in the 1992 sequel, "Honey, I Blew Up the Kid" and appeared as Lisa Barnes in "Where's Rodney?

".
</item 5>

<item 6>
Honey, I Blew Up the Kid
Honey, I Blew Up the Kid is a 1992 American comedy science fiction film and the sequel to the 1989 film "Honey, I Shrunk the Kids".

 Directed by Randal Kleiser and released by Walt Disney Pictures, the film stars Rick Moranis, Marcia Strassman, Robert Oliveri and Amy O'Neill, who reprise
their roles as Wayne, Diane, Nick, and Amy Szalinski respectively, as well as newcomer Keri Russell as Mandy Park, Nick's love interest and babysitter of Adam,
the Szalinskis' new two-year-old son, whose accidental exposure to Wayne's new industrial-sized growth machine causes him to gradually grow to enormous size.

 Made only three years after "Honey, I Shrunk the Kids", this film is set five years after the events depicted in the previous film.

 It was filmed in 1991.
</item 6>

<item 7>
Off His Rockers
Off His Rockers is an animated short film produced by Walt Disney Feature Animation and released in 1992.

 It was released theatrically accompanying the film "Honey, I Blew Up the Kid".

 The film started as a side project of director Barry Cook, who, at the time, was working at Walt Disney Feature Animation Florida.

 Starting with a core of six people, the project ended up involving most of the Florida studio's staff of 73 (and some in California), all of whom "donated"
their time to the project beyond their official duties.
</item 7>

<item 8>
Honey, We Shrunk Ourselves
Honey, We Shrunk Ourselves is a 1997 live-action direct-to-video sequel to "Honey, I Shrunk the Kids" and "Honey, I Blew Up the Kid".

 It is the third and final installment in the "Honey, I Shrunk the Kids" trilogy.

 The directorial debut of cinematographer Dean Cundey and released through Walt Disney Home Video, it tells the story of inventor Wayne Szalinski as he
accidentally shrinks his wife, brother, sister-in-law, and himself with his electromagnetic shrink ray.
</item 8>

<item 9>
The Apple Dumpling Gang Rides Again
The Apple Dumpling Gang Rides Again is a 1979 American comedy-western film produced by Walt Disney Productions and a sequel to "The Apple Dumpling Gang" (1975),
starring the comedy duo of Tim Conway and Don Knotts reprising their respective roles as Amos and Theodore.

 The film also stars Tim Matheson, Harry Morgan, and Kenneth Mars.

 Ruth Buzzi appears in a small cameo as a wild farsighted woman.

 Robert Totten, who directed installments of "Gunsmoke", also had a small part in the film.
</item 9>

<item 10>
Freebie and the Bean
Freebie and the Bean is a 1974 American action-comedy film about two off-beat police detectives who wreak havoc in San Francisco attempting to bring down a
local organized crime boss.

 The picture, a precursor to the buddy cop film genre popularized a decade later, stars James Caan, Alan Arkin, Loretta Swit and Valerie Harper.

 Harper was nominated for the Golden Globe for New Star of the Year for playing the Hispanic wife of Alan Arkin.

 The film was directed by Richard Rush.

 An article in Rolling Stone magazine alleged that Stanley Kubrick called "Freebie and the Bean" the best film of 1974.

 Arkin and Caan would not appear in another movie together until the 2008 film adaptation of "Get Smart".
</item 10>
=== END ITEM id=b8f4d7cd7ba762b4 ===

=== ITEM id=8540b649217e6b7c ===
Gold sentences and facts:
Sentence 0: Semyon Aronovich Gershgorin (August 24, 1901 – May 30, 1933) was a Soviet (born in Pruzhany, Belarus, Russian Empire) mathematician.
  Fact 0.0: Semyon Aronovich Gershgorin was born on August 24, 1901.
  Fact 0.1: Semyon Aronovich Gershgorin died on May 30, 1933.
  Fact 0.2: Semyon Aronovich Gershgorin was Soviet.
  Fact 0.3: Semyon Aronovich Gershgorin was born in Pruzhany, Belarus, Russian Empire.
  Fact 0.4: Semyon Aronovich Gershgorin was a mathematician.
Sentence 1:  He was a mathematician noted for his work on the decidability of various algebraic groups. Malcev algebras (generalisations of Lie algebras) are named after him.
  Fact 1.0: Semyon Aronovich Gershgorin was a mathematician.
  Fact 1.1: Semyon Aronovich Gershgorin was noted for his work on the decidability of various algebraic groups.
  Fact 1.2: Malcev algebras are generalisations of Lie algebras.
  Fact 1.3: Malcev algebras are named after Semyon Aronovich Gershgorin.

Retrieved items:
<item 1>
AES+F
AES+F is an artist collective of four Russian artists: Tatiana Arzamasova (1955), Lev Evzovich (1958), Evgeny Svyatsky (1957), and Vladimir Fridkes (1956).

 It was first formed as AES Group in 1987 by Arzamasova, Evzovich, and Svyatsky, becoming AES+F when Fridkes joined in 1995.

 The collective works in photography, video, installation, and animation, as well as more traditional media, such as painting, drawing, and sculpture.

 AES+F's early work included performance, installation, painting, and illustration.

 Well known for their monumental video-art installations that Gareth Harris describes as "monumental painting set in motion", AES+F create grand visual
narratives that explore contemporary global values, vices and conflicts.
</item 1>

<item 2>
Aleksandr Ivanovsky
Aleksandr Ivanovsky (1881–1968) was a Russian screenwriter and film director who worked during the Soviet era.

 Ivanovsky was awarded the Stalin Prize in 1941, for his work on the 1940 film "Musical Story".

 His 1944 operetta film "Silva" was one of the most popular releases in the Soviet Union that year.
</item 2>

<item 3>
Anatoly Maltsev
Anatoly Ivanovich Maltsev (also: Malcev, Mal'cev; Russian: Анато́лий Ива́нович Ма́льцев; 27 November N.S./14 November O.S. 1909, Moscow Governorate – 7 June
1967, Novosibirsk) was born in Misheronsky, near Moscow, and died in Novosibirsk, USSR.

 He was a mathematician noted for his work on the decidability of various algebraic groups. Malcev algebras (generalisations of Lie algebras) are named after
him.
</item 3>

<item 4>
Semyon Aranovich Gershgorin
Semyon Aronovich Gershgorin (August 24, 1901 – May 30, 1933) was a Soviet (born in Pruzhany, Belarus, Russian Empire) mathematician.

 He began as a student at the Petrograd Technological Institute in 1923, became a Professor in 1930, and was given an appointment at the Leningrad Mechanical
Engineering Institute in the same year.

 His contributions include the Gershgorin circle theorem.
</item 4>

<item 5>
Nicolai V. Krylov
Bellman equations), which were examined with stochastic methods.

 This led to the Evans-Krylov theory, for which he received with Lawrence C. Evans in 2004 the Leroy P. Steele Prize of the American Mathematical Society (for
work done simultaneously and independently by both Krylov and Evans).

 They proved the second order differentiability (Hölder continuity of the second derivative) of the solutions of convex, completely nonlinear, second order
elliptical partial differential equations and thus the existence of "classical solutions" (Theorem of Evans-Krylov).

 He was in 1978 at Helsinki and in 1986 at Berkeley an Invited Speaker for the ICM.

 He received the Humboldt Research Award in 2001.

 In 1993 he was elected a member of the American Academy of Arts and Sciences (1993).

 He should not be confused with the mathematician Nikolay M. Krylov.
</item 5>

<item 6>
Vladimir Arnold
Vladimir Igorevich Arnold (alternative spelling Arnol'd, Russian: Влади́мир И́горевич Арно́льд , 12 June 1937 – 3 June 2010) was a Soviet and Russian
mathematician.

 While he is best known for the Kolmogorov–Arnold–Moser theorem regarding the stability of integrable systems, he made important contributions in several areas
including dynamical systems theory, catastrophe theory, topology, algebraic geometry, symplectic geometry, differential equations, classical mechanics,
hydrodynamics and singularity theory, including posing the ADE classification problem, since his first main result—the solution of Hilbert's thirteenth problem
in 1957 at the age of 19.
</item 6>

<item 7>
Anatoly Fomenko
Anatoly Timofeevich Fomenko (Russian: Анато́лий Тимофе́евич Фоме́нко ) (born 13 March 1945 in Stalino, USSR) is a Soviet and Russian mathematician, professor at
Moscow State University, well known as a topologist, and a member of the Russian Academy of Sciences.

 He is author of a pseudoscientific theory known as New Chronology.

 He is also a member of the Russian Academy of Natural Sciences (1991).
</item 7>

<item 8>
New Chronology (Fomenko)
The New Chronology is a pseudohistorical theory which argues that the conventional chronology of Middle Eastern and European history is fundamentally flawed,
and that events attributed to the civilizations of the Roman Empire, Ancient Greece and Ancient Egypt actually occurred during the Middle Ages, more than a
thousand years later.

 The central concepts of the New Chronology are derived from the ideas of Russian scholar Nikolai Morozov (1854–1946), although work by French scholar Jean
Hardouin (1646–1729) can be viewed as an earlier predecessor.

 However, the New Chronology is most commonly associated with Russian mathematician Anatoly Fomenko (born 1945), although published works on the subject are
actually a collaboration between Fomenko and several other mathematicians.

 The concept is most fully explained in "History: Fiction or Science?"

, originally published in Russian.
</item 8>

<item 9>
Nicolai V. Krylov
Nicolai Vladimirovich Krylov (Russian: Никола́й Влади́мирович Крыло́в ; born 5 June 1941) is a Russian mathematician specializing in partial differential
equations, particularly stochastic partial differential equations and diffusion processes.

 Krylov studied at Lomonosov University, where he in 1966 under E. B. Dynkin attained a doctoral candidate title (similar to a PhD) and in 1973 a Russian
doctoral degree (somewhat more prestigious than a PhD).

 He taught from 1966 to 1990 at the Lomonosov University and is since 1990 a professor at the University of Minnesota.

 At the beginning of his career (starting from 1963) he, in collaboration with Dynkin, worked on nonlinear stochastic control theory, making advances in the
study of convex, nonlinear partial equations of 2nd order ("i.e." Bellman equations), which were examined with stochastic methods.
</item 9>

<item 10>
Anna Aglatova
Anna Khachaturovna Aglatova (Russian: Анна Хачатуровна Аглатова ) is a Russian soprano singer who was born in Kislovodsk and by 2004 joined the singing
department of the Gnessin State Musical College.

 Prior to it, she was a recipient of the "Sergei Leiferkus grant" from the Vladimir Spivakov Fund and in 2005 made her first public appearance at the Bolshoi
Theatre.

 Her career there, have not started with ease since her father didn't want her to work for Bolshoi.

 In 2003, she was a recipient of the first prize at the Bella voce International Competition and the same year participated at both the Christmas Festival at
Düsseldorf and the fourteenth annual Chaliapin Season at Mineralnye Vody.

 She also was a participant for "Irina Arkhipova Fund" and was a part of the Novosibirsk Opera and Ballet Theatre.
</item 10>
=== END ITEM id=8540b649217e6b7c ===

=== ITEM id=da751bdfd2a56fc6 ===
Gold sentences and facts:
Sentence 0: Kolejka ( ; Polish for "queue" or "line (of people)") is a board game designed by Karol Madaj and launched in February 2011 by Poland's Institute of National Remembrance.
  Fact 0.0: Kolejka is a board game.
  Fact 0.1: Kolejka is Polish for 'queue' or 'line (of people)'.
  Fact 0.2: Karol Madaj designed Kolejka.
  Fact 0.3: Kolejka was launched in February 2011.
  Fact 0.4: Kolejka was launched by Poland's Institute of National Remembrance.
Sentence 1: Ghettopoly is a parody of "Monopoly" released in 2003.
  Fact 1.0: Ghettopoly is a parody of Monopoly.
  Fact 1.1: Ghettopoly was released in 2003.

Retrieved items:
<item 1>
Plaid Hat Games
Plaid Hat Games is a United States-based board game studio.

 Plaid Hat Games was founded in 2009.

 Board game designer Colby Dauch formed a board game publishing company in order to release the companies first game, Summoner Wars.
</item 1>

<item 2>
Can't Stop (board game)
Can't Stop is a board game designed by Sid Sackson originally published by Parker Brothers in 1980, and was long out of print in the United States.

 It was reprinted by Face 2 Face Games in 2007.

 An iOS version was developed by Playdek and released in 2012.

 The goal of the game is to "claim" (get to the top of) three of the columns before any of the other players can.

 But the more that the player risks rolling the dice during a turn, the greater the risk of losing the advances made during that turn.
</item 2>

<item 3>
Xeko
Xeko is a collectible card game revolving around endangered species.

 It was launched on Earthday 2006.

 It won the "Creative Child Magazine" 2006 Toy of the Year Award and the National Parenting Center's Seal of Approval in its first year.

 Four "Mission" sets have been released.

 "Mission: Costa Rica" and "Mission: Madagascar", based on biodiversity hotspots were released first.

 "Mission: Indonesia", was released in 2007, with the final release, "Mission: China", was released July 19, 2008.

 A total of thirty more missions were planned but never developed.
</item 3>

<item 4>
Alhambra (board game)
Alhambra (German: Der Palast von Alhambra , literally "The Palace of the Alhambra") is a 2003 tile-based German-style board game designed by Dirk Henn.

 It was originally published in Germany by Queen Games in a language-interdependent version; an English-specific version was released in North America by the
now-defunct Überplay.

 The game is an Arabian-themed update, set during the construction of the Alhambra palace in 14th century Granada, of the 1998 stock trading board game "Stimmt
So!"

, which in turn was an update of the 1992 mafia influence board game "Al Capone"; the original version was subsequently released as "".
</item 4>

<item 5>
Kolejka (game)
Kolejka ( ; Polish for "queue" or "line (of people)") is a board game designed by Karol Madaj and launched in February 2011 by Poland's Institute of National
Remembrance.

 It was designed to teach young generations about the hardships under communism (1945–1989), particularly the difficulty of acquiring consumer goods in the
shortage economy.

 The game has been described as inspired by "Monopoly".
</item 5>

<item 6>
Attack! (board game)
Attack!

 is a board game created by Glenn Drover and published by Eagle Games in 2003.

 It is a light war game that is midway between "Risk" and "Axis and Allies" in complexity.

 The game is loosely set in the 1930s and includes plastic pieces featuring tanks, planes, infantry, and artillery.

 "Attack!"

 won the Origins Award for "Best Historical Game 2003".
</item 6>

<item 7>
Castle Ravenloft Board Game
Castle Ravenloft Board Game is a 2010 board game published by Wizards of the Coast.

 It was the first game released in the Dungeons & Dragons Adventure System board game series.
</item 7>

<item 8>
Ghettopoly
Ghettopoly is a parody of "Monopoly" released in 2003.

 Invented by David Chang, it uses "Monopoly"-like mechanics in the atmosphere of a caricaturized United States ghetto.
</item 8>

<item 9>
Kill Doctor Lucky
Kill Doctor Lucky is a humorous board game designed by James Ernest and released in 1996 by Cheapass Games.

 In 1998, "Kill Doctor Lucky" won the Origins Award for "Best Abstract Board Game of 1997".
</item 9>

<item 10>
Icehouse pieces
Icehouse pieces, or Icehouse Pyramids, Treehouse pieces, Treehouse Pyramids and officially Looney Pyramids, are nestable and stackable pyramid-shaped gaming
pieces and a game system.

 The game system was invented by Andrew Looney and John Cooper in 1987, originally for use in the game of Icehouse.
</item 10>
=== END ITEM id=da751bdfd2a56fc6 ===

=== ITEM id=a37e7b4720ad7049 ===
Gold sentences and facts:
Sentence 0: Not Richard, But Dick is the seventh studio album by The Dead Milkmen, released in 1993 by Hollywood Records.
  Fact 0.0: Not Richard, But Dick is the seventh studio album by The Dead Milkmen.
  Fact 0.1: Not Richard, But Dick was released in 1993.
  Fact 0.2: Not Richard, But Dick was released by Hollywood Records.
Sentence 1: The Dead Milkmen are an American punk rock band formed in 1983 in Philadelphia, Pennsylvania.
  Fact 1.0: The Dead Milkmen are an American punk rock band.
  Fact 1.1: The Dead Milkmen were formed in 1983.
  Fact 1.2: The Dead Milkmen were formed in Philadelphia, Pennsylvania.

Retrieved items:
<item 1>
The J. Geils Band
The J. Geils Band was an American rock band formed in 1968, in Worcester, Massachusetts, under the leadership of guitarist John "J." Geils.

 The other band members included vocalist Peter Wolf, harmonica player Richard "Magic Dick" Salwitz, drummer Stephen Bladd, vocalist/keyboardist Seth Justman
and bassist Danny Klein.

 Wolf and Justman served as principal songwriters.

 The band played R&B-influenced blues rock during the 1970s and soon achieved commercial success before moving towards a more mainstream radio-friendly sound in
the early 1980s, which brought the band to its commercial peak.

 After Wolf left the band in 1983 to pursue a solo career, the band released one more album in 1984 with Justman on lead vocals before breaking up in 1985.

 Beginning in 1999, the band had several reunions prior to the death of its namesake founder J. Geils on April 11, 2017.
</item 1>

<item 2>
Seasoned Veteran
Seasoned Veteran is the third studio album by American rapper Richie Rich.

 It was released November 5, 1996 on Def Jam Recordings, and to date is the only major label release for the artist.

 The album was produced by Ali Malik, DJ Daryl, Doug Rasheed, Jermaine Dupri, Lev Berlak, Mike Mosley, Richie Rich and Rick Rock.

 It peaked at number 11 on the "Billboard" Top R&B/Hip-Hop Albums and at number 35 on the "Billboard" 200.

 Two of the singles released, "Let's Ride" and "Do G's Get To Go To Heaven?

", both appeared on multiple "Billboard" singles charts.

 The album features guest performances by 2Pac, E-40, Luniz, D'wayne Wiggins, T-Boz, and Rame Royal of Rhythm & Green.
</item 2>

<item 3>
Richie Furay
Paul Richard "Richie" Furay (born May 9, 1944) is an American singer, songwriter, and Rock & Roll Hall of Fame member who is best known for forming the bands
Buffalo Springfield with Stephen Stills, Neil Young, Bruce Palmer, and Dewey Martin, and Poco with Jim Messina, Rusty Young, George Grantham and Randy Meisner.

 His best known song (originally written during his tenure in Buffalo Springfield, but eventually performed by Poco, as well) was "Kind Woman", which he wrote
for his wife, Nancy.
</item 3>

<item 4>
The Dead Milkmen
The Dead Milkmen are an American punk rock band formed in 1983 in Philadelphia, Pennsylvania.

 Their original lineup consisted of vocalist and keyboardist Rodney Linderman ("Rodney Anonymous"), guitarist and vocalist Joe Genaro ("Joe Jack Talcum"),
bassist Dave Schulthise ("Dave Blood") and drummer Dean Sabatino ("Dean Clean").
</item 4>

<item 5>
The V.I.P.s (band)
The V.I.P.s were a British R&B musical ensemble formed in Carlisle, Cumberland, (North West England) in late 1963, out of an earlier outfit known as The
Ramrods, who had formed in Carlisle in 1960.

 After a change of personnel in April 1967, the band changed their name to "Art", and released the album "Supernatural Fairy Tales".

 They also participated to a psychedelic bizarre album called "Featuring The Human Host And The Heavy Metal Kids" by a collective known as Hapshash and the
Coloured Coat, formed by Guy Stevens and an influential British graphic design and avant-garde musical partnership between Michael English and Nigel Waymouth.

 The musicians involved in that project were Mike Harrison on keys and vocals, Luther Grosvenor on guitars, Greg Ridley on bass and Mike Kellie on drums, as
well as performances by Stevens, English and Waymouth.

 It was the first time that the term "heavy metal" was ever used in music, even though that album had nothing to do with heavy metal music, being closer to
psychedelic music.
</item 5>

<item 6>
Electric Guitars
Electric Guitars were an English band formed early in 1980 by Neil Davenport (vocals, lyrics) and Richard Hall (bass, vocals) who were both studying English at
Bristol University.

 The band soon increased to a five-man line-up, with Andy Saunders (guitar, vocals), Matt Salt (drums) and Dick Truscott (keyboards), they also later added two
backing singers: Sara and Wendy Patridge.

 Their first single "Health" / "Continental Shelf" was released on local label Fried Egg Records in 1980.
</item 6>

<item 7>
RPA &amp; The United Nations of Sound
The band played in Australia also during the Splendour in the Grass Festival in late July.

 The band's debut album as Richard's backing lineup, entitled "United Nations of Sound", was released on 19 July 2010.

 "Born Again", the first proper single, was officially released the same day of the album as a digital download.
</item 7>

<item 8>
Richard Cheese
Richard Cheese & Lounge Against the Machine is a cover band and comedy act, performing popular songs in a lounge/swing style, reminiscent of Frank Sinatra, Dean
Martin, and Tony Bennett.

 Lounge singer Richard Cheese is a character created and portrayed by Los Angeles-based actor/comedian Mark Jonathan Davis.
</item 8>

<item 9>
Not Richard, But Dick
Not Richard, But Dick is the seventh studio album by The Dead Milkmen, released in 1993 by Hollywood Records.

 After being out of print for years, Hollywood Records decided to release the album for digital download as of April 2, 2013.
</item 9>

<item 10>
Dick Delicious and The Tasty Testicles
Dick Delicious and The Tasty Testicles are a comedy / metal band from Atlanta, Georgia.

 The band formed in 1992 and released 3 albums between 1995 and 2003.

 In 2003 the Dick Delicious received the "Howard Stern Award for Musical Excellence" and broke up shortly afterward.
</item 10>
=== END ITEM id=a37e7b4720ad7049 ===

=== ITEM id=09bbd592d3b32d0c ===
Gold sentences and facts:
Sentence 0:  The group included: Alexander Bogdanov, Maxim Gorky, Anatoly Lunacharsky, Mikhail Pokrovsky, Grigory Aleksinsky, Stanislav Volski, and Martyn Liadov.
  Fact 0.0: The group included Alexander Bogdanov.
  Fact 0.1: The group included Maxim Gorky.
  Fact 0.2: The group included Anatoly Lunacharsky.
  Fact 0.3: The group included Mikhail Pokrovsky.
  Fact 0.4: The group included Grigory Aleksinsky.
  Fact 0.5: The group included Stanislav Volski.
  Fact 0.6: The group included Martyn Liadov.
Sentence 1: Anatoly Vasilyevich Lunacharsky (Russian: Анато́лий Васи́льевич Лунача́рский , , 23 November [O.S. 11 November] 1875 – 26 December 1933) was a Russian Marxist revolutionary and the first Soviet People's Commissar of Education responsible for culture and education.
  Fact 1.0: Anatoly Vasilyevich Lunacharsky's Russian name is Анато́лий Васи́льевич Лунача́рский.
  Fact 1.1: Anatoly Vasilyevich Lunacharsky was born on 23 November 1875 (Old Style 11 November 1875).
  Fact 1.2: Anatoly Vasilyevich Lunacharsky died on 26 December 1933.
  Fact 1.3: Anatoly Vasilyevich Lunacharsky was a Russian Marxist revolutionary.
  Fact 1.4: Anatoly Vasilyevich Lunacharsky was the first Soviet People's Commissar of Education.
  Fact 1.5: Anatoly Vasilyevich Lunacharsky was responsible for culture and education as People's Commissar of Education.

Retrieved items:
<item 1>
Apollinariya Yakubova
Apollinariya Yakubova (died 1913 or 1917) was a Russian Marxist revolutionary and, with Vladimir Lenin, one of the founders of the League of Struggle for the
Emancipation of the Working Class.
</item 1>

<item 2>
Vladimir Bazarov
Vladimir Alexandrovich Bazarov ( Влади́мир Алекса́ндрович База́ров; 1874–1939) was a Russian Marxist revolutionary, journalist, philosopher, and economist, born
Vladimir Alexandrovich Rudnev.

 Bazarov is best remembered as a pioneer in the development of economic planning in the Soviet Union.
</item 2>

<item 3>
Martemyan Ryutin
Martemyan Nikitich Ryutin (Russian: Мартемья́н Ники́тич Рю́тин ) (1890–1937) was a Russian Marxist revolutionary, and a political functionary of the Russian
Communist Party.

 Ryutin is best remembered as the leader of a pro-peasant political faction organized against Soviet leader Joseph Stalin in the early 1930s and as the primary
author of a 200-page oppositional platform.

 Ryutin was arrested by the Soviet secret police, along with his co-thinkers, in what has come to be known as the Ryutin Affair.

 He was executed in January 1937 as part of the Yezhovshchina (Great Purge) conducted against political oppositionists and suspected economic "wreckers" and
spies.
</item 3>

<item 4>
Petrashevsky Circle
The Petrashevsky Circle was a Russian literary discussion group of progressive-minded commoner-intellectuals in St. Petersburg in the 1840s.

 It was organized by Mikhail Petrashevsky, a follower of the French utopian socialist Charles Fourier.

 Among the members were writers, teachers, students, minor government officials, and army officers.

 While differing in political views, most of them were opponents of the tsarist autocracy and Russian serfdom.

 Like that of the Lyubomudry group founded earlier in the century, the purpose of the circle was to discuss Western philosophy and literature that was
officially banned by the Imperial government of Tsar Nicholas I.
</item 4>

<item 5>
You Fell Victim to a Fateful Struggle
You Fell Victim (Russian: "Вы жертвою пали"; "Vy zhertvoiu pali"), or You Fell Victim to a Fateful Struggle, is a Russian Marxist and revolutionary funeral
march.

 It acted as the funeral dirge of the Russian revolutionary movement, among them the Bolsheviks.
</item 5>

<item 6>
The entries profile Russian and Soviet figures across science, arts, politics, and activism: entomologist and prolific author Pavel Marikovsky; film director
Aleksandr Ivanovsky; animator Vladimir Danilevich; actors and collectors Ekaterina and Vladimir Semenikhin; and actor Aleksei Petrenko. They also describe the
anti-tsarist Petrashevsky Circle and member Nikolay Speshnev, plus Marxist revolutionaries, thinkers, and officials including Bazarov, Bag
</item 6>

<item 7>
Vpered
Vpered ("Forward" or "Hasten") (1909 - 1912) was an organization emanating from within the Russian Social Democratic Labour Party (Russian Social Democracy or
RSDLP).

 The faction was gathered by Alexander Bogdanov (1873 - 1928) in December 1909.

 The group included: Alexander Bogdanov, Maxim Gorky, Anatoly Lunacharsky, Mikhail Pokrovsky, Grigory Aleksinsky, Stanislav Volski, and Martyn Liadov.

 Although Vpered emerged from the Bolshevik wing of Russian Social Democracy, the group was critical of Lenin.
</item 7>

<item 8>
Aleksandra Sokolovskaya
Aleksandra Lvovna Sokolovskaya (Russian: Александра Львовна Соколовская ; 1872 – c. 1938) was a Russian Marxist revolutionary and Leon Trotsky's first wife.

 She perished in the Great Purges no earlier than 1938.
</item 8>

<item 9>
Matvey Skobelev
Matvey Ivanovich Skobelev (Russian: Матве́й Ива́нович Ско́белев ; November 9, 1885, Baku – July 29, 1938, Moscow) was a Russian Marxist revolutionary and
politician.
</item 9>

<item 10>
Nikolay Speshnev
Nikolay Alexandrovich Speshnev (Russian: Николай Александрович Спе́шнев; 1821, Kursk - 1882, St. Petersburg) was a 19th-century Russian aristocrat and political
activist, best known for his involvement with the pro-socialist literary discussion group the Petrashevsky Circle.

 He formed a secret revolutionary society from among the members of the circle, which included the young Fyodor Dostoevsky.

 After the government of Tsar Nicholas I arrested the members of the Petrashevsky Circle in 1849, Speshnev was interrogated, threatened with torture, and
eventually sentenced, along with Dostoevsky, Petrashevsky and others, to execution by firing squad.

 The sentence was commuted to hard labour in Siberia, but the prisoners were only informed of this after enduring a mock execution.
</item 10>
=== END ITEM id=09bbd592d3b32d0c ===

=== ITEM id=e1f9bb772640f68b ===
Gold sentences and facts:
Sentence 0: "Grow Some Funk of Your Own" is a song by Elton John.
  Fact 0.0: 'Grow Some Funk of Your Own' is a song by Elton John.
Sentence 1: Sir Elton Hercules John, (born Reginald Kenneth Dwight; 25 March 1947), is an English singer, pianist, and composer.
  Fact 1.0: Sir Elton Hercules John was born Reginald Kenneth Dwight.
  Fact 1.1: Sir Elton Hercules John was born on 25 March 1947.
  Fact 1.2: Sir Elton Hercules John is an English singer.
  Fact 1.3: Sir Elton Hercules John is an English pianist.
  Fact 1.4: Sir Elton Hercules John is an English composer.
Sentence 2:  John owned Watford Football Club from 1976 to 1987, and 1997 to 2002.
  Fact 2.0: Elton John owned Watford Football Club from 1976 to 1987.
  Fact 2.1: Elton John owned Watford Football Club from 1997 to 2002.

Retrieved items:
<item 1>
A Boy in a Man's World
A Boy in a Man's World is the second studio album by Mucky Pup.

 The album was recorded at Nevessa Productions and released in 1989 through Torrid Records and Roadrunner Records.

 The album features a re-recorded version of "U-Stink-But-I-♥-U," which the band had previously won a Bloom County songwriting contest with.

 The band would also shoot their first music video for this song.

 The album also features "Batman", a song based on an urban legend that also contains a snippet of the title theme to the 1966 "Batman" TV.
</item 1>

<item 2>
Firm Biz
"Firm Biz" is the first single by the Hip hop supergroup The Firm from their singular collaborative LP "The Album".

 The song was produced by L.E.S, who based the song's track on a sample of Teena Marie's 1981 hit "Square Biz".

 "Firm Biz" also features a chorus performed by former En Vogue singer Dawn Robinson, reworked from the "Square Biz" chorus.

 AZ raps the first verse of the song, Nas the second, and Foxy Brown the third and final verse.

 The song never reached any US chart positions, but was a UK Top 20 hit in 1997.

 A remix of the song was released, featuring a verse from Half-A-Mill and a chorus by Mary J. Blige.

 The word "feds" is censored in the explicit version, in verse 2, performed by Nas, when he says "Never that though Black .4-4's for feds".
</item 2>

<item 3>
Elton John
He has also composed music, produced records, and has occasionally acted in films.

 John owned Watford Football Club from 1976 to 1987, and 1997 to 2002.

 He is an honorary Life President of the club, and in 2014 had a stand named after him at the club's home stadium.
</item 3>

<item 4>
Knitting Factory Records
Knitting Factory Records is an independent American music label that is notable for promoting a variety of artists, including the music of deceased Nigerian
political activist Fela Kuti.

 The label promotes a variety of music artists including Ages and Ages, Ash Black Bufflo, Cuong Vu, Graham Haynes, Femi Kuti, Gary Lucas, Lumerians, Thomas
Chapin, Patrolled By Radar, Joe Morris, Rachid Taha, Seun Kuti, and Shilpa Ray and her Happy Hookers.
</item 4>

<item 5>
Wood Green Town F.C.
Wood Green Town Football Club was a football club based in Tottenham in the London Borough of Haringey, England.

 Formed in 1911 as a breakaway from Tufnell Park, they merged with Edmonton (a descendent of Tufnell Park) in 1973 to form Edmonton & Haringey.

 During their history the club were known as Tufnell Spartans, Wood Green, Wood Green Town and Haringey Borough.
</item 5>

<item 6>
Blitz Kids
The Blitz Kids were a group of young people who frequented the weekly Blitz club-night in Covent Garden, London in 1979-80, and are credited with launching the
New Romantic subcultural movement.

 It was Andrew Czezowski (Ex manager of The Damned) and Susan Carrington who went on to start The Roxy club with Barry Jones that introduced Steve Strange and
Rusty Egan to the Blitz in Covent Garden, which started the New Romantic Movement.

 Among the many attendees were Rusty Egan, Steve Strange, Boy George, Marilyn, Alice Temple, Perri Lister, Princess Julia, Philip Sallon, Carl Teper and Martin
Degville (later to be the frontman of Tony James' Sigue Sigue Sputnik).

 Crucially, the club lay between two art colleges (St Martin's School and Central School) and became a testbed for student fashion designers who set London
ablaze during the 1980s.
</item 6>

<item 7>
The Happenings
The Happenings are a pop music group that originated in the 1960s.

 Members of the original group, created in the spring of 1961 and initially called "The Four Grads" because all had just graduated from high school in Paterson,
New Jersey, were Bob Miranda, David Libert, Tom Giuliano, and Ralph DiVito.

 In 1968 DiVito was replaced by Bernie LaPorta and Lenny Conforti also joined to play drums in the touring band.

 Both LaPorta and Conforti took a hiatus from the northern New Jersey band, The Emerald Experience, to play and tour with The Happenings.

 That lineup performed mostly at colleges and universities until 1970, when Libert left the band to manage other groups, including George Clinton and
Parliament-Funkadelic, Living Colour, Brian Auger, Vanilla Fudge, The Runaways (Cherie Currie, Joan Jett, Lita Ford), Mother's Finest, Alice Cooper and Evelyn
"Champagne" King.
</item 7>

<item 8>
Afrodiziak
Afrodiziak was a British singing group composed of Caron Wheeler, Claudia Fontaine, and later Naomi Thompson, that was active in the 1980s.

 As a duo, Wheeler and Fontaine were best known for performing backing vocals on The Jam's final single "Beat Surrender" in 1982 (with whom they also performed
on their final tour) and Elvis Costello's 1983 album "Punch the Clock", especially its single, the international hit "Everyday I Write the Book".

 After the addition of a third member, Naomi Thompson, they went on to sing for acts such as Heaven 17 and Howard Jones.

 Afrodiziak performed backing vocals on the hit single Free Nelson Mandela, including the a cappella intro, staging it at The Tube (Channel4) in March 1984.

 Heaven 17's 1984 album, "How Men Are", featured them prominently, especially on the singles "Sunset Now" and "And That's No Lie".
</item 8>

<item 9>
Grow Some Funk of Your Own
"Grow Some Funk of Your Own" is a song by Elton John.

 It was released as a single in 1976 and comes from his album "Rock of the Westies".

 It shared its A-side status with "I Feel Like a Bullet (In the Gun of Robert Ford)".

 The song is considered to be one of John's heavier rock songs.

 The song went to #14 on the "Billboard" Hot 100, but in Britain broke a five-year run of successful singles by failing to reach the Top 50 despite extensive
radio play.

 Although the singles chart listed only "Funk" and not "Bullet", the latter song was given equal billing in later "Billboard" singles chart books.
</item 9>

<item 10>
South Liverpool F.C.
South Liverpool Football Club is a football club based in Liverpool, England, founded as a phoenix club of a club of the same name.

 They are currently members of the West Cheshire League Division One and play at Jericho Lane in Otterspool.

 The club's colours are white shirts, black shorts and red socks.
</item 10>
=== END ITEM id=e1f9bb772640f68b ===

=== ITEM id=49c9f6a788b34fd7 ===
Gold sentences and facts:
Sentence 0:  Although called a "city", Qionghai refers to a large land area in Hainan - an area which was once a county.
  Fact 0.0: Qionghai is called a city.
  Fact 0.1: Qionghai refers to a large land area in Hainan.
  Fact 0.2: The area of Qionghai in Hainan was once a county.
Sentence 1:  It has an area of 1692 km2 and in 2010, it had a population of 483,217.
  Fact 1.0: Qionghai has an area of 1692 km2.
  Fact 1.1: Qionghai had a population of 483,217 in 2010.
Sentence 2: Suining (; Sichuanese Pinyin: Xunin; Sichuanese pronunciation: ; ) is a prefecture-level city of eastern Sichuan province in Southwest China.
  Fact 2.0: Suining has the Sichuanese Pinyin Xunin.
  Fact 2.1: Suining is a prefecture-level city.
  Fact 2.2: Suining is in eastern Sichuan province.
  Fact 2.3: Sichuan province is in Southwest China.
Sentence 3:  In 2002, Suining had a population of 658,798.
  Fact 3.0: Suining had a population of 658,798 in 2002.

Retrieved items:
<item 1>
Xingcheng
Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban
inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.

 Currently under the administration of Huludao City, the area is steeped in history, and contains one of the best preserved Ming Dynasty towns in China, as well
as functioning as a laidback summer resort.
</item 1>

<item 2>
Qinghe District, Tieling
Qinghe District () is a district of Tieling, Liaoning province, People's Republic of China.

 However, it is located closer to Kaiyuan City than it is to downtown Tieling, which lies 39 km to the southwest.
</item 2>

<item 3>
Longkou
Longkou (), formerly Huang County (), is a port city in northeastern Shandong province, China, facing the Bohai Sea to the north and the Laizhou Bay to the
west.

 Longkou, a county-level city, is administered by the prefecture-level city of Yantai.

 The total population of Longkou is 620,000.
</item 3>

<item 4>
Ezhou
Ezhou () is a prefecture-level city in eastern Hubei Province, China.

 As of the 2010 census, the city had a population of 1,048,668, of which 668,727 lived in the core Echeng District.

 The Ezhou - Huanggang built-up ("or metro") area was home to 1,035,496 inhabitants from the Echeng and Huangzhou, Huanggang Districts.
</item 4>

<item 5>
Kaiyuan, Liaoning
Kaiyuan () is a county-level city in the northeast of Liaoning, People's Republic of China, bordering Jilin for a small section to the north.

 It is under the administration of Tieling City, the centre of which lies 33 km to the southwest.
</item 5>

<item 6>
Daying County
Daying County () is a county of Sichuan province, China, under the administration of Suining City and in the central part of the Sichuan Basin.

 In 2002, it had a population of 520,000 residing in an area of 703 km2 .

 Daying is remarkable for being the home of the "Chinese Dead Sea", a tourist attraction that is based on a salt lake with 9 times the salinity of the ocean and
which constitutes the largest indoor water park in China.

 The G42 Shanghai–Chengdu Expressway and Dazhou–Chengdu Railway (达成铁路) traverses east–west through the entire length of the county.
</item 6>

<item 7>
Qionghai
Qionghai () is one of the seven county-level cities of Hainan province, China.

 Although called a "city", Qionghai refers to a large land area in Hainan - an area which was once a county.

 Within this area is the main city, Qionghai City.

 It is located in the east of the island at the mouth of the Wanquan River, 86 km from the provincial capital of Haikou.

 It has an area of 1692 km2 and in 2010, it had a population of 483,217.
</item 7>

<item 8>
Dujiangyan City
Dujiangyan () is a county-level city, a subdivision of Chengdu, the capital of Sichuan Province, People's Republic of China.

 Its north-west region forms a border with sourthern Ngawa Tibetan and Qiang Autonomous Prefecture.

 It has an area of 1208 km2 and had a population of 600,000 in 2003.
</item 8>

<item 9>
Suining
Suining (; Sichuanese Pinyin: Xunin; Sichuanese pronunciation: ; ) is a prefecture-level city of eastern Sichuan province in Southwest China.

 In 2002, Suining had a population of 658,798.
</item 9>

<item 10>
Dongtai
Dongtai () is a coastal county-level city under the administration of Yancheng, Jiangsu province, China.

 It has a population of roughly 1,170,000 estimated for 2007.

 Out of the total population, about 260,000 live in the Dongtai urban core, others are distributed in the 23 suburban towns and rural regions (Some famous towns
include Touzao, Anfeng, Qingdong, Sancang, Qianggang, Fu'an, Tangyang, etc.).

 With some Yellow Sea coast, it borders the prefecture-level cities of Nantong to the south and Taizhou to the west, and is the southernmost county-level
division of Yancheng.
</item 10>
=== END ITEM id=49c9f6a788b34fd7 ===

=== ITEM id=3324a72012047f33 ===
Gold sentences and facts:
Sentence 0: The Edison Sault Power Canal supplies the Saint Marys Falls Hydropower Plant, a Cloverland Electric Cooperative hydroelectric plant, in Sault Ste. Marie, Michigan.
  Fact 0.0: The Edison Sault Power Canal supplies the Saint Marys Falls Hydropower Plant.
  Fact 0.1: The Saint Marys Falls Hydropower Plant is a Cloverland Electric Cooperative hydroelectric plant.
  Fact 0.2: The Saint Marys Falls Hydropower Plant is in Sault Ste. Marie, Michigan.
Sentence 1:  Excavation of the power canal began in September 1898 and was completed in June 1902.
  Fact 1.0: Excavation of the Edison Sault Power Canal began in September 1898.
  Fact 1.1: Excavation of the Edison Sault Power Canal was completed in June 1902.
Sentence 2:  The canal and hydroelectric complex were named a Historic Civil Engineering Landmark in 1983.
  Fact 2.0: The canal was named a Historic Civil Engineering Landmark in 1983.
  Fact 2.1: The hydroelectric complex was named a Historic Civil Engineering Landmark in 1983.
Sentence 3: The Cayuga–Seneca Canal is a canal in New York, United States.
  Fact 3.0: The Cayuga–Seneca Canal is a canal in New York, United States.
Sentence 4:  It is now part of the New York State Canal System.
  Fact 4.0: The Cayuga–Seneca Canal is now part of the New York State Canal System.

Retrieved items:
<item 1>
Cayuga–Seneca Canal
The Cayuga–Seneca Canal is a canal in New York, United States.

 It is now part of the New York State Canal System.
</item 1>

<item 2>
New York State Canal System
The New York State Canal System (formerly known as the New York State Barge Canal) is a successor to the Erie Canal and other canals within New York.

 Currently, the 525 mi system is composed of the Erie Canal, the Oswego Canal, the Cayuga–Seneca Canal, and the Champlain Canal.

 In 2014 the system was listed as a national historic district on the National Register of Historic Places in its entirety, and in 2016 it was designated a
National Historic Landmark.
</item 2>

<item 3>
New York State Canal Corporation
The New York State Canal Corporation is responsible for the oversight, administration and maintenance of the New York State Canal System, which consists of the
Erie Canal, Cayuga–Seneca Canal, Oswego Canal and Champlain Canal.

 It is also involved with the development and maintenance of the New York State Canalway Trail and with the general development and promotion of the Erie Canal
Corridor as both a tourist attraction and a working waterway.
</item 3>

<item 4>
Seneca River Crossing Canals Historic District
Seneca River Crossing Canals Historic District is a national historic district located at Montezuma and Tyre in Cayuga and Seneca Counties, New York.

 The district includes more than a mile of the Enlarged Erie Canal prism (built here between 1849 and 1857); towpath and heelpath; a drydock; the remains of the
Richmond (Montezuma) Aqueduct crossing the Seneca River; remnants of the original Erie Canal, built between 1817 and 1825 and including Lock #62 and piers of
the original mule bridge from that era; and a culvert that carries a stream beneath the Enlarged Erie Canal.
</item 4>

<item 5>
Power canal
A Power Canal refers to a canal used for hydraulic power generation, rather than for transport of watercraft.

 The power canal was a major factor in the Industrial revolution in New England in the 19th century.

 Most early power canals were mill races used mechanically to transfer power directly from falling water to machinery in mill buildings.

 Later, the hydraulic power generated electricity locally for the same mill factories.

 These power canals were often filled in as electricity (transported by power lines) replaced the need for local water power, and road transport needs or city
expansion needs reclaimed the land.

 Some hydraulic power canals were transformed into local electric generators, but most were closed.

 Remains of power canals can be seen in old mill towns and are often protected as historical structures today.
</item 5>

<item 6>
Beaver and Erie Canal
The Beaver and Erie Canal, also known as the Erie Extension Canal, was part of the Pennsylvania Canal system and consisted of three sections: the Beaver
Division, the Shenango Division, and the Conneaut Division.

 The canal ran 136 mi north–south near the western edge of the state from the Ohio River to Lake Erie through Beaver County, Lawrence County, Mercer County,
Crawford County, and Erie County, Pennsylvania.
</item 6>

<item 7>
Raritan Water Power Canal
The Raritan Water Power Canal in Raritan, New Jersey was a three-mile long water power canal built during the early 1840s to power industries.

 The canal has been determined to be eligible for the National Register.

 The canal is included in Duke Island Park, as part of the Somerset County Park System.
</item 7>

<item 8>
Lowell Power Canal System and Pawtucket Gatehouse
The Lowell Power Canal System is the largest power canal system in the United States, at 5.6 miles in length.

 The system's estimated output is 10,000 horsepower, operating six major canals on two levels, controlled by numerous gates.

 The system was begun in the 1790s, beginning its life as a transportation canal called the Pawtucket Canal, which was constructed to get logs from New
Hampshire down the Merrimack River to shipbuilding centers at Newburyport, Massachusetts, bypassing the 30-plus-foot drop of the Pawtucket Falls.
</item 8>

<item 9>
Dundee Canal
The Dundee Canal was an industrial canal in Clifton and Passaic in Passaic County, New Jersey.

 It was built between 1858 and 1861 and ran parallel to the Passaic River.

 It supplied hydropower and water for manufacturing.

 There was interest by some members of the business community to modify the canal to support navigational uses, but the canal was never used for that purpose.
</item 9>

<item 10>
Edison Sault Power Canal
The Edison Sault Power Canal supplies the Saint Marys Falls Hydropower Plant, a Cloverland Electric Cooperative hydroelectric plant, in Sault Ste. Marie,
Michigan.

 Excavation of the power canal began in September 1898 and was completed in June 1902.

 The canal and hydroelectric complex were named a Historic Civil Engineering Landmark in 1983.
</item 10>
=== END ITEM id=3324a72012047f33 ===

=== ITEM id=e954fc2a75243b1a ===
Gold sentences and facts:
Sentence 0: Hydrangea ( ; common names hydrangea or hortensia) is a genus of 70–75 species of flowering plants native to southern and eastern Asia (China, Japan, Korea, the Himalayas, and Indonesia) and the Americas.
  Fact 0.0: Hydrangea is a genus of flowering plants.
  Fact 0.1: Hydrangea's common names are hydrangea or hortensia.
  Fact 0.2: The Hydrangea genus has 70 to 75 species.
  Fact 0.3: Hydrangea plants are native to southern and eastern Asia.
  Fact 0.4: Hydrangea plants are native to China, Japan, Korea, the Himalayas, and Indonesia.
  Fact 0.5: Hydrangea plants are native to the Americas.
Sentence 1:  It contains, 13–18 species, but is best known for a single species, Jequirity ("A. precatorius").
  Fact 1.0: The subject of this sentence contains 13 to 18 species.
  Fact 1.1: The subject of this sentence is best known for a single species.
  Fact 1.2: That single species is Jequirity.
  Fact 1.3: Jequirity is also called A. precatorius.

Retrieved items:
<item 1>
Hydrangea quercifolia
Hydrangea quercifolia, commonly known by its translation oakleaf hydrangea or oak-leaved hydrangea, is a species of flowering plant native to the Southeastern
United States, in woodland habitats from North Carolina west to Tennessee, and south to Florida and Louisiana.

 A deciduous shrub with white showy flower heads, it is a commonly grown garden plant.

 Numerous cultivars are available commercially.
</item 1>

<item 2>
Hydrangea serrata
Hydrangea serrata is a species of flowering plant in the family Hydrangeaceae, native to mountainous regions of Korea and Japan.

 Common names include mountain hydrangea and tea of heaven.

 Growing to 1.2 m tall and broad, it is a deciduous shrub with oval leaves and panicles of blue and pink flowers in summer and autumn (fall).

 It is widely cultivated as an attractive ornamental shrub throughout the world in areas with suitable climate and soil.
</item 2>

<item 3>
Hydrangea arborescens
Hydrangea arborescens, commonly known as smooth hydrangea, wild hydrangea, or sevenbark, is a species of flowering plant in the family Hydrangeaceae.

 It is a small- to medium-sized, deciduous shrub up to 3 m tall that is native to the eastern United States.
</item 3>

<item 4>
Hydrangea paniculata
Hydrangea paniculata, the panicled hydrangea, is a species of flowering plant in the family Hydrangeaceae native to southern and eastern China, Korea, Japan and
Russia (Sakhalin).

 It was first formally described by Philipp Franz von Siebold in 1829.

 It is a deciduous shrub or small tree, 1 - tall by 2.5 m broad, growing in sparse forests or thickets in valleys or on mountain slopes.
</item 4>

<item 5>
Hydrangea macrophylla
Hydrangea macrophylla is a species of flowering plant in the family Hydrangeaceae, native to Japan.

 It is a deciduous shrub growing to 2 m tall by 2.5 m broad with large heads of pink or blue flowers in summer and autumn.

 Common names include bigleaf hydrangea, French hydrangea, lacecap hydrangea, mophead hydrangea, penny mac and hortensia.

 It is widely cultivated in many parts of the world in many climates.

 It is not to be confused with "H. aspera" 'Macrophylla'.
</item 5>

<item 6>
Deutzia
Deutzia ( or ) is a genus of about 60 species of flowering plants in the family Hydrangeaceae, native to eastern and central Asia (from the Himalayas east to
Japan and the Philippines), and Central America and also Europe.

 By far the highest species diversity is in China, where 50 species occur.
</item 6>

<item 7>
Abrus canescens
Abrus canescens is a species of flowering plant belonging to the legume family, native to Africa.

 It is considered almost extinct.
</item 7>

<item 8>
Hydrangea heteromalla
Hydrangea heteromalla is a species of "Hydrangea" flower, native to the Himalayas and China.

 Average height is 10'-15' with inflorescences that are 5"-8" across and bloom May-June,full flowering only occurs in full sun.Is a very adaptable plant that is
hardy to USDA zones (4)5-7.

 This plant is seldom seen in American gardens although there is nice specimen at the Hillier Arboretum.
</item 8>

<item 9>
Hydrangea
Hydrangea ( ; common names hydrangea or hortensia) is a genus of 70–75 species of flowering plants native to southern and eastern Asia (China, Japan, Korea, the
Himalayas, and Indonesia) and the Americas.

 By far the greatest species diversity is in eastern Asia, notably China, Japan, and Korea.

 Most are shrubs 1 to 3 meters tall, but some are small trees, and others lianas reaching up to 30 m by climbing up trees.

 They can be either deciduous or evergreen, though the widely cultivated temperate species are all deciduous.
</item 9>

<item 10>
Abrus
Abrus is a genus of flowering plants in the pea family, Fabaceae and the only genus found in the tribe Abreae.

 It contains, 13–18 species, but is best known for a single species, Jequirity ("A. precatorius").

 The highly toxic seeds of that species are used to make jewellery.
</item 10>
=== END ITEM id=e954fc2a75243b1a ===

=== ITEM id=9da697050cc736aa ===
Gold sentences and facts:
Sentence 0: Julia Compton Moore (February 10, 1929April 18, 2004) was the wife of Lieutenant General (Ret.)
  Fact 0.0: Julia Compton Moore's dates are February 10, 1929 to April 18, 2004.
  Fact 0.1: Julia Compton Moore was the wife of Lieutenant General (Ret.).
Sentence 1:  Hal Moore, a United States Army officer.
  Fact 1.0: Hal Moore is a United States Army officer.
Sentence 2: Harold Gregory "Hal" Moore, Jr. (February 13, 1922 – February 10, 2017) was a United States Army lieutenant general and author.
  Fact 2.0: Harold Gregory "Hal" Moore, Jr. lived from February 13, 1922 to February 10, 2017.
  Fact 2.1: Harold Gregory "Hal" Moore, Jr. was a United States Army lieutenant general.
  Fact 2.2: Harold Gregory "Hal" Moore, Jr. was an author.
Sentence 3:  He was a recipient of the Distinguished Service Cross, which is the U.S. military's second highest decoration for valor, and was the first of his West Point class (1945) to be promoted to brigadier general, major general, and lieutenant general.
  Fact 3.0: Hal Moore was a recipient of the Distinguished Service Cross.
  Fact 3.1: The Distinguished Service Cross is the U.S. military's second highest decoration for valor.
  Fact 3.2: Hal Moore was the first of his West Point class (1945) to be promoted to brigadier general.
  Fact 3.3: Hal Moore was the first of his West Point class (1945) to be promoted to major general.
  Fact 3.4: Hal Moore was the first of his West Point class (1945) to be promoted to lieutenant general.

Retrieved items:
<item 1>
Andrew Davis Bruce
Lieutenant General Andrew Davis Bruce (September 14, 1894 – July 28, 1969) was an American academic and soldier who served as the third president of the
University of Houston.

 He retired from the United States Army in 1954 as a lieutenant general after seeing action in both World War I and World War II and founding Fort Hood, Texas.

 Three countries, France, the Philippines, and the United States, awarded him service medals, including the Distinguished Service Cross, the U.S. Army's second
highest military decoration.

 Bruce is interred in Arlington National Cemetery.
</item 1>

<item 2>
Hal Moore
Harold Gregory "Hal" Moore, Jr. (February 13, 1922 – February 10, 2017) was a United States Army lieutenant general and author.

 He was a recipient of the Distinguished Service Cross, which is the U.S. military's second highest decoration for valor, and was the first of his West Point
class (1945) to be promoted to brigadier general, major general, and lieutenant general.
</item 2>

<item 3>
Matthew Ridgway
General Matthew Bunker Ridgway (March 3, 1895 – July 26, 1993) was the 19 Chief of Staff of the United States Army.

 He served with great distinction during World War II, where he was the Commanding General (CG) of the 82nd Airborne Division, leading it in action in Sicily,
Italy and Normandy, before taking command of the newly formed XVIII Airborne Corps in August 1944, holding this post until the end of the war, commanding it in
the Battle of the Bulge, Operation Varsity and the Western Allied invasion of Germany.
</item 3>

<item 4>
Wayne W. Lambert
Wayne W. Lambert (born 1936) was an Air force Brigadier General (United States).

 He attended the United States Military Academy at West Point, New York, after already having served a year in the US Army.

 Lambert graduated from the Academy in 1959 and was commissioned a 2nd Lieutenant in the USAF that same year.

 He trained as a bomber pilot, receiving his Aviator badge in 1960.

 He has logged over 5,800 flying hours with the US Air Force.

 A seasoned combat veteran, Lambert flew over 225 combat missions in South East Asia as a B-52 crew member.

 He was awarded the Distinguished Flying Cross (United States) and received 10 awards of the Air Medal for his service during the Vietnamese War.

 After serving in a succession of lesser command positions, Lambert was promoted to the rank of Brigadier General in 1983, assuming command of the former
Strategic Air Command’s 7th Air Division, headquartered then at Ramstein Air Base in West Germany, and overseeing SAC operations throughout the European
theater.

 The Brigadier General retired from the US Air Force in 1989.
</item 4>

<item 5>
Richard E. Cavazos
Richard Edward Cavazos (born January 31, 1929), a Korean War recipient of the Distinguished Service Cross as a first lieutenant, who advanced in rank to become
the United States Army's first Hispanic four-star general.

 During the Vietnam War, as a lieutenant colonel, Cavazos was awarded a second Distinguished Service Cross.

 In 1976, Cavazos became the first Mexican American to reach the rank of brigadier general in the U.S. Army.

 Cavazos served with great distinction for thirty-three years, with his final command as head of the U.S. Army Forces Command.
</item 5>

<item 6>
John F. Anderson
John Francis Anderson was an officer in the Union Army during the American Civil War.

 When the war broke out, Anderson who was the son of Maine's former Governor Hugh J. Anderson was initially commissioned First Lieutenant in the militia; and on
September 2, 1861, Anderson became the adjutant of the 24th Regiment Massachusetts Volunteer Infantry.

 He was appointed to Maj. Gen. John G. Foster's staff as a major and an aide-de-camp on June 9, 1863 and served with that officer for most of the remainder of
the war, resigning on March 27, 1865.

 On December 11, 1866, President Andrew Johnson nominated Major Anderson to the honorary grade of brevet brigadier general, to rank from March 13, 1865 and the
United States Senate confirmed the award on February 6, 1867.
</item 6>

<item 7>
Nadja West
Nadja Y. West (born 1961) is a United States Army lieutenant general and the 44th U.S. Army Surgeon General and Commanding General of the U.S. Army Medical
Command.

 West is the first black Army Surgeon General, and was the first black female, active-duty, major general and the first black female major general in Army
Medicine.

 West is also the first Army black female lieutenant general.

 She is the highest ranking female to have graduated from the United States Military Academy.

 She received the Army Distinguished Service Medal, the Defense Superior Service Medal, the Legion of Merit, and others.
</item 7>

<item 8>
Julia Compton Moore
Julia Compton Moore (February 10, 1929April 18, 2004) was the wife of Lieutenant General (Ret.)

 Hal Moore, a United States Army officer.

 Her efforts and complaints in the aftermath of the Battle of Ia Drang prompted the U.S. Army to set up survivor support networks and casualty notification
teams consisting of uniformed officers, which are still in use.
</item 8>

<item 9>
Distinguished Service Cross (United States)
The Distinguished Service Cross is the second highest military award that can be given to a member of the United States Army (and previously, the United States
Army Air Forces and the United States Air Force), for extreme gallantry and risk of life in actual combat with an armed enemy force.

 Actions that merit the Distinguished Service Cross must be of such a high degree that they are above those required for all other U.S. combat decorations but
do not meet the criteria for the Medal of Honor.

 The Distinguished Service Cross is equivalent to the Navy Cross (Navy and Marine Corps), the Air Force Cross (Air Force), and the Coast Guard Cross (Coast
Guard).
</item 9>

<item 10>
Mark E. Mitchell
Mark E. Mitchell is the current Acting Assistant Secretary of Defense for Special Operations and Low-Intensity Conflict of the United States Department of
Defense.

 A retired Colonel, Mitchell was the first member of the United States Army to be awarded the Distinguished Service Cross during the War in Afghanistan and was
the first to receive the award since the Vietnam War.

 In 2003, he received the award for his actions during the Battle of Qala-i-Jangi, which took place in late November – early December 2001.
</item 10>
=== END ITEM id=9da697050cc736aa ===
