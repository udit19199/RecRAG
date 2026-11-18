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

This file has 11 items.

=== ITEM id=ed88cac1f63437e8 ===
Gold sentences and facts:
Sentence 0:  It is the third of Gluck's so-called reform operas for Vienna, following "Alceste" and "Orfeo ed Euridice", and the least often performed of the three.
  Fact 0.0: The work is the third of Gluck's so-called reform operas for Vienna.
  Fact 0.1: The work follows 'Alceste' and 'Orfeo ed Euridice' among Gluck's reform operas for Vienna.
  Fact 0.2: The work is the least often performed of the three reform operas for Vienna.
Sentence 1: Orfeo ed Euridice (French version: Orphée et Eurydice ; English: "Orpheus and Eurydice") is an opera composed by Christoph Willibald Gluck based on the myth of Orpheus, set to a libretto by Ranieri de' Calzabigi.
  Fact 1.0: Orfeo ed Euridice is an opera composed by Christoph Willibald Gluck.
  Fact 1.1: Orfeo ed Euridice is based on the myth of Orpheus.
  Fact 1.2: Orfeo ed Euridice is set to a libretto by Ranieri de' Calzabigi.
  Fact 1.3: The French version of Orfeo ed Euridice is titled Orphée et Eurydice.
  Fact 1.4: The English title of Orfeo ed Euridice is 'Orpheus and Eurydice'.
Sentence 2:  The piece was first performed at the Burgtheater in Vienna on 5 October 1762 in the presence of Empress Maria Theresa.
  Fact 2.0: The piece was first performed at the Burgtheater in Vienna on 5 October 1762.
  Fact 2.1: The first performance of the piece was in the presence of Empress Maria Theresa.

Retrieved items:
<item 1>
Gottlieb Muffat
Gottlieb Muffat (April 1690 – 9 December 1770), son of Georg Muffat, served as "Hofscholar" under Johann Fux in Vienna from 1711 and was appointed to the
position of third court organist at the "Hofkapelle" in 1717.

 He acquired additional duties over time including the instruction of members of the Imperial family, among them the future Empress Maria Theresa.

 He was promoted to second organist in 1729 and first organist upon the accession of Maria Theresa to the throne in 1741.

 He retired from official duties at the court in 1763.
</item 1>

<item 2>
Paride ed Elena
Paride ed Elena (] ; "Paris and Helen") is an opera by Christoph Willibald Gluck.

 It is the third of Gluck's so-called reform operas for Vienna, following "Alceste" and "Orfeo ed Euridice", and the least often performed of the three.

 Like its predecessors, the libretto was written by Ranieri de' Calzabigi.

 The opera tells the story of the events between the Judgment of Paris and the flight of Paris and Helen to Troy.

 It was premiered at the Burgtheater in Vienna on 3 November 1770.
</item 2>

<item 3>
Romolo ed Ersilia (Hasse)
Romolo ed Ersilia is an opera in three acts composed by Johann Adolph Hasse to an Italian-language libretto by Pietro Metastasio.

 The opera was commissioned by Empress Maria Theresa to celebrate the marriage of her son Leopold to Maria Luisa of Spain.

 The opera was first performed on 6 August 1765, the day after their wedding, in the court theatre of the Imperial Palace in Innsbruck.

 The opera received its first performance in modern times in 2011 during the Innsbruck Festival of Early Music.
</item 3>

<item 4>
Bernhard Bötel
His stage repertoire included Belmonte in Wolfgang Amadeus Mozart's "Die Entführung aus dem Serail", Chapelou in Adolphe Adam's "Le postillon de Lonjumeau",
Count Almaviva in Gioachino Rossini's "The Barber of Seville", Daniel in Franz Lehár's "The Merry Widow", the Duke of Mantua in Giuseppe Verdi's "Rigoletto",
Gabriel von Eisenstein in Johann Strauss II's "Die Fledermaus", Indigo in Strauss' "Indigo und die vierzig Räuber", Jeník in Bedřich Smetana's "The Bartered
Bride", Paolino in Domenico Cimarosa's "Il matrimonio segreto", Pâris in Jacques Offenbach's "La belle Hélène", Pietro in Franz von Suppé's "Boccaccio", and
Wilhelm Meister in Ambroise Thomas' "Mignon".
</item 4>

<item 5>
Orfeo ed Euridice
Orfeo ed Euridice (French version: Orphée et Eurydice ; English: "Orpheus and Eurydice") is an opera composed by Christoph Willibald Gluck based on the myth of
Orpheus, set to a libretto by Ranieri de' Calzabigi.

 It belongs to the genre of the "azione teatrale", meaning an opera on a mythological subject with choruses and dancing.

 The piece was first performed at the Burgtheater in Vienna on 5 October 1762 in the presence of Empress Maria Theresa.

 "Orfeo ed Euridice" is the first of Gluck's "reform" operas, in which he attempted to replace the abstruse plots and overly complex music of "opera seria" with
a "noble simplicity" in both the music and the drama.
</item 5>

<item 6>
Operetta (film)
Operetta (German: Operette) is a 1940 musical film directed by Willi Forst and starring Forst, Maria Holst and Dora Komar.

 The film was made by Wien-Film, a Vienna-based company set up after Austria had been incorporated into Greater Germany following the 1938 Anschluss.

 It is the first film in director Willi Forst's "Viennese Trilogy" followed by "Vienna Blood" (1942) and "Viennese Girls" (1945).

 The film portrays the life of Franz Jauner (1832–1900), a leading musical figure in the city.

 It is both an operetta film and a Wiener Film.
</item 6>

<item 7>
Die glückliche Hand
Die glückliche Hand ("The Hand of Fate"), Op.

 18, is a "Drama mit Musik" ("drama with music") by Arnold Schoenberg in four scenes.

 It was composed between 1910 and 1913.

 Like "Erwartung", composed a year earlier, it was heavily influenced by Otto Weininger's book "Sex and Character".

 Unlike "Erwartung", Schoenberg wrote the libretto for "Die glückliche Hand" himself.

 The first performance took place in Vienna on 24 October 1924.

 The underlying message of the piece is the idea that man continues to repeatedly make the same mistakes, and the plot is developed from events in Schoenberg’s
personal life.
</item 7>

<item 8>
Daphne (opera)
Daphne, Op.

82, is an opera in one act by Richard Strauss, subtitled "Bucolic Tragedy in One Act".

 The German libretto was by Joseph Gregor.

 The opera is based loosely on the mythological figure Daphne from Ovid's "Metamorphoses" and includes elements taken from "The Bacchae" by Euripides.

 The opera premiered at the Semperoper in Dresden on 15 October 1938, originally intended as a double bill with Strauss' "Friedenstag", but as the scale of
"Daphne" grew, that idea was abandoned.

 The conductor of the first performance was Karl Böhm, to whom the opera was dedicated.
</item 8>

<item 9>
Maria Theresa (film)
Maria Theresa (German: Maria Theresia) is a 1951 Austrian historical drama film directed by  Emil E. Reinert and starring  Paula Wessely, Fred Liewehr and
Marianne Schönauer.

 It portrays the life of the eighteenth century Habsburg Empress Maria Theresa.
</item 9>

<item 10>
Das geheime Königreich
Das geheime Königreich ("The Secret Kingdom") is an opera in one act with words and music by Ernst Krenek, his Op.

 50 and the second of three one-acters (with "Der Diktator" and "Schwergewicht, oder Die Ehre der Nation") which premiered on 6 May 1928 at the Hessisches
Staatstheater Wiesbaden as part of the Mai-Festwoche Wiesbaden.

 It is subtitled "Märchenoper in 1 Akt " (fairy-tale opera in one act) and has been called a satirical fairy-tale opera.
</item 10>
=== END ITEM id=ed88cac1f63437e8 ===

=== ITEM id=8cce8afdef5b3fb9 ===
Gold sentences and facts:
Sentence 0: Mr. Wilson's Cabinet of Wonder: Pronged Ants, Horned Humans, Mice on Toast, and Other Marvels of Jurassic Technology is a book by Lawrence Weschler primarily about the Museum of Jurassic Technology and, more broadly, the history and role of museums.
  Fact 0.0: Mr. Wilson's Cabinet of Wonder: Pronged Ants, Horned Humans, Mice on Toast, and Other Marvels of Jurassic Technology is a book.
  Fact 0.1: Lawrence Weschler wrote Mr. Wilson's Cabinet of Wonder.
  Fact 0.2: Mr. Wilson's Cabinet of Wonder is primarily about the Museum of Jurassic Technology.
  Fact 0.3: Mr. Wilson's Cabinet of Wonder is more broadly about the history and role of museums.
Sentence 1: The Museum of Jurassic Technology is a museum located at 9341 Venice Boulevard in the Palms district of Los Angeles, California.
  Fact 1.0: The Museum of Jurassic Technology is a museum.
  Fact 1.1: The Museum of Jurassic Technology is located at 9341 Venice Boulevard.
  Fact 1.2: The Museum of Jurassic Technology is in the Palms district of Los Angeles, California.

Retrieved items:
<item 1>
Imperial War Museum (disambiguation)
The Imperial War Museum is a British national museum organisation with branches at five locations in England.
</item 1>

<item 2>
Museum of Jurassic Technology
The Museum of Jurassic Technology is a museum located at 9341 Venice Boulevard in the Palms district of Los Angeles, California.

 It was founded by David Hildebrand Wilson and Diana Drake Wilson (husband and wife) in 1988.
</item 2>

<item 3>
Dudley Zoo
Dudley Zoological Gardens is a 40 acre zoo located within the grounds of Dudley Castle in the town of Dudley, in the Black Country region of the West Midlands,
England.

 The Zoo opened to the public on 18 May 1937.

 Dudley Zoo is owned and operated by Dudley and West Midlands Zoological Society, founded in 1935 and a registered charity.
</item 3>

<item 4>
Rice Northwest Museum of Rocks and Minerals
The Rice Northwest Museum of Rocks and Minerals is a non-profit museum in Hillsboro, Oregon, United States.

 Located just north of the Sunset Highway on the northern edge of Hillsboro, the earth science museum is in the Portland metropolitan area.

 Opened in 1997, the museum’s collections date to the 1930s with the museum housed in a home built to display the rock and mineral collections of the museum
founders.

 The ranch style home is listed on the National Register of Historic Places, the first of its kind listed in Oregon.

 In 2015 the museum became a Smithsonian Affiliate museum.
</item 4>

<item 5>
The Lost City (Dungeons &amp; Dragons)
The Lost City (B4) is a "Dungeons & Dragons" adventure module by Tom Moldvay.

 It was first published by TSR in 1982 and was designed as a stand-alone adventure for use with the "Dungeons & Dragons Basic Set".

 The working title for the module was "The Lost City of Cynidecia".

 Moldvay designed the module to give novice Dungeon Masters experience fleshing out adventures and is only partially complete.

 The module is described as a low-level scenario, in which the only hope of the player characters' survival can be found in a ruined city slowly rising out of
the sands.

 The adventure is set inside a huge step pyramid, with the lower pyramid only sketched out and the city itself described with a list of the major areas and a
map.

 The adventure’s main villain is Zargon, a giant one-eyed monster and his minions.

 The entire double pyramid, not including the city, contains over 100 rooms.
</item 5>

<item 6>
Witmer Stone
Witmer Stone (September 22, 1866 – May 24, 1939) was an American ornithologist, botanist, and mammalogist, and was considered one of the last of the “great
naturalists.”

 Stone is remembered principally as an ornithologist.

 He was president of the American Ornithologists’ Union (AOU) 1920–23, and was editor of the AOU’s periodical "The Auk" 1912–1936.

 He spearheaded the production of the 4th edition of the AOU checklist, published in 1931.

 He worked for over 50 years in the Ornithology Department at the Academy of Natural Sciences of Philadelphia, eventually serving as Director of the
institution.

 Stone was one of the founding members of the Delaware Valley Ornithological Club (DVOC) in 1890 and was actively involved in the organization for the remainder
of his life.

 Stone was one of only two scientists (Joseph Grinnell was the other) to serve as president of both the AOU and the American Society of Mammalogists, and he
co-authored two popular books about mammals.
</item 6>

<item 7>
Primeval Whirl
Primeval Whirl is a steel wild mouse roller coaster at Disney's Animal Kingdom at the Walt Disney World Resort.

 The ride is a spinning roller coaster purchased from Reverchon Industries.

 The ride is part of Chester and Hester's Dino-Rama, which is itself part of DinoLand U.S.A. It is a roller coaster in the "Mild But Wild Thrills" category.

 It has cars that spin in circles while traveling on tracks, permitting the ride experience to vary greatly each time it is ridden.

 The ride features 13 cars, each seating up to 4 riders.
</item 7>

<item 8>
Mr. Wilson's Cabinet of Wonder
Mr. Wilson's Cabinet of Wonder: Pronged Ants, Horned Humans, Mice on Toast, and Other Marvels of Jurassic Technology is a book by Lawrence Weschler primarily
about the Museum of Jurassic Technology and, more broadly, the history and role of museums.
</item 8>

<item 9>
David Hildebrand Wilson
David Hildebrand Wilson is the co-founder, along with his wife, Diana Wilson, of the enigmatic Museum of Jurassic Technology in Los Angeles, California.

 After high school, Wilson enrolled at Kalamazoo College where he majored in urban entomology with a minor in art.

 He received an MFA in Experimental Animation from the California Institute of the Arts in 1976.

 He was awarded the MacArthur Fellowship and the Creative Capital Moving Image Award in 2001.

 He and his museum are the subject of the book, "Mr. Wilson's Cabinet of Wonder", published in 1995.
</item 9>

<item 10>
Merlin's Wood
Merlin's Wood; or, The Vision of Magic is a short novel by British writer Robert Holdstock, first published in the United Kingdom in 1994.

 The novel is considered part of the "Mythago Wood" cycle, but takes place in Brittany, France instead of Herefordshire, England.

 The work has all new characters and focuses on the mythical birthplace and burial site of Merlin, the magical wood Brocéliande.

 Brocéliande is a smaller version of Ryhope wood where British myth predominates.
</item 10>
=== END ITEM id=8cce8afdef5b3fb9 ===

=== ITEM id=c034278042f639ff ===
Gold sentences and facts:
Sentence 0: August or Auguste Brancart (21 July 1851 - 1894?)
  Fact 0.0: August Brancart is also known as Auguste Brancart.
  Fact 0.1: August Brancart was born on 21 July 1851.
  Fact 0.2: The end date given for August Brancart is 1894, with a question mark.
Sentence 1:  He also published work of the Decadent movement such as "Monsieur Vénus" by Rachilde.
  Fact 1.0: Brancart published work of the Decadent movement.
  Fact 1.1: Brancart published "Monsieur Vénus" by Rachilde.
  Fact 1.2: "Monsieur Vénus" by Rachilde is an example of Decadent movement work.
Sentence 2: The Decadent Movement was a late 19th-century artistic and literary movement, centered in Western Europe, that followed an aesthetic ideology of excess and artificiality.
  Fact 2.0: The Decadent Movement was an artistic and literary movement.
  Fact 2.1: The Decadent Movement was a late 19th-century movement.
  Fact 2.2: The Decadent Movement was centered in Western Europe.
  Fact 2.3: The Decadent Movement followed an aesthetic ideology of excess.
  Fact 2.4: The Decadent Movement followed an aesthetic ideology of artificiality.

Retrieved items:
<item 1>
The Rebel (book)
The Rebel (French: "L'Homme révolté" ) is a 1951 book-length essay by Albert Camus, which treats both the metaphysical and the historical development of
rebellion and revolution in societies, especially Western Europe.

 Camus relates writers and artists as diverse as Epicurus and Lucretius, Marquis de Sade, Georg Wilhelm Friedrich Hegel, Fyodor Dostoyevsky, Friedrich
Nietzsche, Max Stirner, André Breton, and others in an integrated, historical portrait of man in revolt.

 Examining both rebellion and revolt, which may be seen as the same phenomenon in personal and social frames, Camus examines several 'countercultural' figures
and movements from the history of Western thought and art, noting the importance of each in the overall development of revolutionary thought and philosophy.
</item 1>

<item 2>
Hermann Brunn
Karl Hermann Brunn (1 August 1862 – 20 September 1939) was a German mathematician, known for his work in convex geometry (see Brunn–Minkowski inequality) and in
knot theory.

 Brunnian links are named after him, as his 1892 article "Über Verkettung" included examples of such links.
</item 2>

<item 3>
Decadent movement
The Decadent Movement was a late 19th-century artistic and literary movement, centered in Western Europe, that followed an aesthetic ideology of excess and
artificiality.

 The visual artist Félicien Rops's body of work and Joris-Karl Huysmans's novel "Against Nature" (1884) are considered the prime examples of the decadent
movement.

 It first flourished in France and then spread throughout Europe and to the United States.

 The movement was characterized by self-disgust, sickness at the world, general skepticism, delight in perversion and employment of crude humor and a belief in
the superiority of human creativity over logic and the natural world.
</item 3>

<item 4>
Christian de la Mazière
Christian de la Mazière (August 22, 1922 in Tunis – February 15, 2006) was a journalist and member of the Charlemagne Division of the Waffen SS.

 He is known for discussing his role in the documentary "The Sorrow and the Pity" and also wrote a book titled "The Captive Dreamer".

 At the start of the war, he served in the French Army and was later retained in the military of Vichy France until 1942.

 After being discharged, he worked for the fascist newspaper "Le Pays Libre", joining the Charlemagne Division just before the Liberation of Paris in 1944.

 He was taken prisoner in Pomerania by Polish forces in the Red Army.
</item 4>

<item 5>
Marcel J. Melançon
Marcel J. Mélançon is a Canadian philosopher and scientist.

 He was born in Saint-Barnabé, Quebec, in 1938, and is mostly known for his book on the French philosopher Albert Camus "Albert Camus, An Analysis of his
thought" (Albert Camus.

 Analyse de sa pensée).

 He has mostly worked together with the biologist Richard D. Lambert.
</item 5>

<item 6>
Auguste Brancart
August or Auguste Brancart (21 July 1851 - 1894?)

 was a Belgian publisher of pornographic literature, credited with the first publication of "My Secret Life".

 He published translations of English pornography into French and vice versa for English publishers such as Edward Avery.

 He also published work of the Decadent movement such as "Monsieur Vénus" by Rachilde.
</item 6>

<item 7>
Herbert Lottman
Herbert Lottman (August 16, 1927, Brooklyn - August 27, 2014, Paris) was an American author who specialized in writing biographies on French subjects.

 An influential biographer, he published 17 biographies, 15 of which were related to French culture, commerce, or politics; including works on Albert Camus,
Colette, Gustave Flaubert, Henri Philippe Pétain, Jules Verne, and the Rothschild banking family of France.

 He wrote that, just before dying, Albert Camus was pledged to marry.

 Camus's estate tried to block his book, partly because of this controversial statement.
</item 7>

<item 8>
Albert Camus
Albert Camus (] ; 7 November 1913 – 4 January 1960) was a French philosopher, author, and journalist.

 His views contributed to the rise of the philosophy known as absurdism.

 He wrote in his essay "The Rebel" that his whole life was devoted to opposing the philosophy of nihilism while still delving deeply into individual freedom.

 He won the Nobel Prize in Literature in 1957.
</item 8>

<item 9>
Donald Allen
Donald Merriam Allen (Iowa, 1912 – San Francisco, August 29, 2004) was an influential editor, publisher, and translator of contemporary American literature.

 He is perhaps best known for his project "The New American Poetry 1945-1960" (1960), among the several important anthologies of contemporary American
innovative writing he made available to the public.

 Allen began his working life as a Japanese translator within the US military, serving in WWII.

 After his military service, Allen became an editor at Grove Press, where he worked for sixteen years.

 He was one of the first translators of the Romanian-French Absurdist playwright Eugène Ionesco, and Allen's 1958 volume "Four Plays of Eugène Ionesco" helped
to introduce the playwright to American audiences in the 1960s.
</item 9>

<item 10>
Cult MTL
Cult MTL is an English language arts, culture and news website and monthly print publication, based in Montreal, Quebec, Canada.

 Its first print edition appeared on 7 September 2012.

 It was created only a few months after Montreal's last English-language alternative weekly, "Montreal Mirror", was unceremoniously closed by its parent
company, Quebecor.

 The founding editor's of "Cult MTL" were also involved with the "Mirror".

 In August 2013 the print version of the magazine was started.
</item 10>
=== END ITEM id=c034278042f639ff ===

=== ITEM id=5ac7da74f0b86fbe ===
Gold sentences and facts:
Sentence 0: Eenasul Fateh (Bengali: ঈনাসুল ফাতেহ ; born 3 April 1959), also known by his stage name Aladin, is a Bangladeshi-British cultural practitioner, magician, live artist and former international management consultant.
  Fact 0.0: Eenasul Fateh was born on 3 April 1959.
  Fact 0.1: Eenasul Fateh's Bengali name is ঈনাসুল ফাতেহ.
  Fact 0.2: Eenasul Fateh is also known by his stage name Aladin.
  Fact 0.3: Eenasul Fateh is a Bangladeshi-British cultural practitioner.
  Fact 0.4: Eenasul Fateh is a magician.
  Fact 0.5: Eenasul Fateh is a live artist.
  Fact 0.6: Eenasul Fateh is a former international management consultant.
Sentence 1: Management consulting is the practice of helping organizations to improve their performance, operating primarily through the analysis of existing organizational problems and the development of plans for improvement.
  Fact 1.0: Management consulting is the practice of helping organizations improve their performance.
  Fact 1.1: Management consulting operates primarily through the analysis of existing organizational problems.
  Fact 1.2: Management consulting operates primarily through the development of plans for improvement.

Retrieved items:
<item 1>
Management consulting
Management consulting is the practice of helping organizations to improve their performance, operating primarily through the analysis of existing organizational
problems and the development of plans for improvement.

 Organizations may draw upon the services of management consultants for a number of reasons, including gaining external (and presumably objective) advice and
access to the consultants' specialized expertise.
</item 1>

<item 2>
Alex Algard
Alex Algard is an Internet entrepreneur.

 He is the founder and former CEO of CarDomain, the founder and former CEO of Whitepages.com, and the founder and current CEO of Hiya.
</item 2>

<item 3>
Allen Klein
Allen Klein (December 18, 1931 July 4, 2009) was an American businessman, music publisher, writers' representative, filmmaker and record label executive, most
noted for his tough persona and aggressive, innovative negotiation tactics, many of which established higher industry standards for compensating recording
artists.

 He founded ABKCO Music & Records Incorporated.

 Klein revolutionized the income potential of recording artists, who previously had been routinely victimized by onerous record company contracts.

 He first scored massive monetary and contractual windfalls for Buddy Knox and Jimmy Bowen, one-hit rockabillies of the late 1950s, then parlayed his early
successes into a position managing Sam Cooke, and eventually managed the Beatles and the Rolling Stones simultaneously, along with many other artists, becoming
one of the most powerful individuals in the music industry during his era.
</item 3>

<item 4>
Da'ud Abu al-Fadl
Da'ud Abu al-Fadl (1161–1242) was a Karaite Jewish physician who lived in the Ayyubid Egypt in the twelfth century CE.

 He born in Cairo in 1161 and died there about 1242.

 Having studied medicine under the Jewish physician Hibat Allah ibn Jami, and under Abu al-Fafa'il ibn Naqid, he became the court physician of the sultan
al-Malik al-'Adil Abu Bakr ibn Ayyub, the brother and successor of Saladin.

 He was also chief professor at the al-Nasiri Hospital at Cairo, where he had a great many pupils, among them being the historian ibn Abi Usaibiyyah.

 The latter declared that Abu al-Fadl was the most skillful physician of the time and that his success in curing the sick was miraculous.

 Abu al-Fadl was the author of an Arabic pharmacopoeia in twelve chapters, entitled "Aḳrabadhin", and treating chiefly of antidotes.
</item 4>

<item 5>
Alan Kamil
Alan C. "Al" Kamil is an American experimental psychologist.

 He is the Director, School of Biological Sciences and George Holmes Professor of Biological Sciences and Psychology at the University of Nebraska, Lincoln.

 Kamil's work focusses on the evolution of memory and adaptive specializations of learning in many animal species, especially the Clark's nutcracker and other
birds.

 Kamil has published peer reviewed articles on both theoretical aspects of comparative psychology and animal cognition, and on empirical studies of animal
learning and memory.

 In 2013 Kamil was honoured by the Comparative Cognition Society for his contributions to the study of animal cognition.
</item 5>

<item 6>
Eenasul Fateh
Eenasul Fateh (Bengali: ঈনাসুল ফাতেহ ; born 3 April 1959), also known by his stage name Aladin, is a Bangladeshi-British cultural practitioner, magician, live
artist and former international management consultant.
</item 6>

<item 7>
Abdal
Abdāl ("lit": substitutes, but which can also mean "generous" ["karīm"] and "noble" ["sharīf"]) is a term used in Islamic metaphysics and Islamic mysticism,
both Sunni and Shiite, to refer to a particularly important group of God's saints.

 In the tradition of Sunni Islam in particular, the concept attained an especially important position in the writings of the Sunni mystics and theologians,
whence it appears in the works of Sunni authorities as diverse as Abu Talib al-Makki (d. 956), Ali Hujwiri (d. 1072), Ibn Asakir (d. 1076), Khwaja Abdullah
Ansari (d. 1088), and Ibn Khaldun (d. 1406).
</item 7>

<item 8>
Donald Allen
Donald Merriam Allen (Iowa, 1912 – San Francisco, August 29, 2004) was an influential editor, publisher, and translator of contemporary American literature.

 He is perhaps best known for his project "The New American Poetry 1945-1960" (1960), among the several important anthologies of contemporary American
innovative writing he made available to the public.

 Allen began his working life as a Japanese translator within the US military, serving in WWII.

 After his military service, Allen became an editor at Grove Press, where he worked for sixteen years.

 He was one of the first translators of the Romanian-French Absurdist playwright Eugène Ionesco, and Allen's 1958 volume "Four Plays of Eugène Ionesco" helped
to introduce the playwright to American audiences in the 1960s.
</item 8>

<item 9>
Aladdin's Eatery
Aladdin's Eatery is a chain of franchised restaurants in the American Midwest and South-East, specializing in Lebanese cuisine.

 Adapted to American tastes, the sites are fast casual restaurants that also offer take out.
</item 9>

<item 10>
Abd Allah ibn al-Fadl
Little is known of his life, apart from what can be gleaned from manuscripts of his texts.

 He was a deacon and the grandson of a bishop.

 He received an excellent education in both Arabic and Greek, having studied Arabic grammar with the famous poet Abul ʿAla Al-Maʿarri.

 Several of his works and translations were commissioned by notables from Antioch and neighbouring cities in Muslim territory.
</item 10>
=== END ITEM id=5ac7da74f0b86fbe ===

=== ITEM id=d6880f63d44b7abf ===
Gold sentences and facts:
Sentence 0: During World War II, Operation Cold Comfort was a failed SAS raid that began with a parachute drop north of Verona on February 17, 1945.
  Fact 0.0: Operation Cold Comfort was a failed SAS raid.
  Fact 0.1: Operation Cold Comfort took place during World War II.
  Fact 0.2: Operation Cold Comfort began with a parachute drop.
  Fact 0.3: The parachute drop of Operation Cold Comfort was north of Verona.
  Fact 0.4: The parachute drop of Operation Cold Comfort was on February 17, 1945.
Sentence 1:  The SAS was founded in 1941 as a regiment, and later reconstituted as a corps in 1950.
  Fact 1.0: The SAS was founded in 1941 as a regiment.
  Fact 1.1: The SAS was later reconstituted as a corps.
  Fact 1.2: The SAS was reconstituted as a corps in 1950.

Retrieved items:
<item 1>
Cold Comfort (Inside No. 9)
"Cold Comfort" is the fourth episode of the second series of the British dark comedy anthology television programme "Inside No. 9".

 The episode, which was written and directed by Steve Pemberton and Reece Shearsmith, was first broadcast on 16 April 2015 on BBC Two.

 Most of "Cold Comfort" is composed of a stream from a fixed camera on the desk of Andy, the protagonist, with smaller pictures on the side of the screen, in
the style of a CCTV feed.

 "Cold Comfort" was filmed over two and a half days in Twickenham, and was, like "A Quiet Night In" from "Inside No. 9"'s first series, highly experimental.

 It was Pemberton and Shearsmith's directorial debut.
</item 1>

<item 2>
Jan Breytenbach
Jan Dirk Breytenbach {'1': ", '2': ", '3': ", '4': "} (born 4 July 1933) was appointed by General Fritz Loots, the founder of the South African Special Forces
Brigade, as the first commander of 1 Reconnaissance Commando, the first unit founded within the South African Special Forces.

 He was also appointed as the first commander of the 32 Battalion, known colloquially as "Buffalo Battalion", as well as 44 Parachute Brigade.
</item 2>

<item 3>
The passage profiles special forces worldwide: the Philippine Army’s airborne regiment trains with U.S. Green Berets; Britain’s SAS conducts reconnaissance,
counterterrorism, direct action, and hostage rescue; Singapore’s SOF serves in the SOTF; Japan’s WWII Giretsu paratroopers targeted Allied air raids; Chile’s
Iquique commandos belong to an armored brigade; Jordan’s 1963-founded forces perform intelligence, rescue, counterterrorism, and precision strikes; South
African Jan B
</item 3>

<item 4>
Orlov let
Orlov let ("Eagle's Flight") was the name of an anti-terrorist operation conducted by the Montenegrin police to arrest a group of Albanians who planned
terrorist attacks and an armed conflict in Albanian-inhabited parts of Montenegro.

 The group of 17 people planned operations from 2004 to their arrest on the night of 10/11 September 2006.

 Weapons and explosive material were found in police raids.

 The group's main operative was the destroying of cultural and religious buildings (Orthodox) in Albanian-inhabited territory.

 The group was organised by an Albanian association based in Detroit, and members of the Kosovo Liberation Army, from which they received financial aid and
smuggled arms into Montenegro.

 The group was given a total sentence of 51 years of prison.

 Relations between the state and the Albanian minority strained after the event.
</item 4>

<item 5>
Special Air Service
The Special Air Service (SAS) is a special forces unit of the British Army.

 The SAS was founded in 1941 as a regiment, and later reconstituted as a corps in 1950.

 The unit undertakes a number of roles including covert reconnaissance, counter-terrorism, direct action and hostage rescue.
</item 5>

<item 6>
Joint Special Operations Command (Jordan)
The Special Operation Forces of the Jordanian Armed Forces serve as Jordan's premiere special forces unit.

 Founded on April 15, 1963 on the orders of the late King Hussein, its primary roles include reconnaissance, counter-terrorism, search and evacuation,
intelligence gathering combat, and the protection of key sites.

 The Special Operation Forces are also charged with carrying out precision strikes against critical enemy targets.

 The 14,000-strong unit are equipped and trained to be able to operate behind enemy lines for long periods without any logistical support, and is considered one
of the finest special forces units in the world.
</item 6>

<item 7>
Operation Barras
Operation Barras was a British Army operation that took place in Sierra Leone on 10 September 2000, during the late stages of that nation's civil war.

 The operation aimed to release five British soldiers of the Royal Irish Regiment who had been held by a militia group known as the "West Side Boys".

 The soldiers were part of a patrol that was returning from a visit to Jordanian peacekeepers attached to the United Nations Mission in Sierra Leone (UNAMSIL)
at Masiaka on 25 August 2000 when they turned off the main road and down a track towards the village of Magbeni.

 There the patrol was overwhelmed by a large number of heavily armed rebels, taken prisoner, and transported to Gberi Bana on the opposite side of Rokel Creek.
</item 7>

<item 8>
Special Operations Force (Singapore)
The Special Operations Force (SOF) is the Republic's Army Special Forces composed of highly trained elite soldiers within the Singapore Armed Forces Commando
Formation and an essential component of the joint special forces unit Special Operations Task Force (SOTF).

 According to the Principles of Special Forces, the Republic's special forces thrive on their exceptional qualities and advanced skills, consisting highly
adaptive individuals who are independent and can operate independently, and the Republic's special forces soldiers cannot be mass-produced and must be managed
carefully.
</item 8>

<item 9>
Operation Cold Comfort
During World War II, Operation Cold Comfort was a failed SAS raid that began with a parachute drop north of Verona on February 17, 1945.

 It was later renamed Zombie.
</item 9>

<item 10>
The entries cover Allied operations across World War II and one ancient battle: the 1943 invasion of Italy (Salerno, Calabria, Taranto); Normandy’s 1944 Epsom
offensive and Gold Beach landings; the 21st Army Group’s campaign in northwest Europe; and the 1945 Allied advance into Germany and southern Germany. Other
events include Operation Barras’s 2000 rescue of British hostages in Sierra Leone, the deadly 1943 Naples post-office bombing
</item 10>
=== END ITEM id=d6880f63d44b7abf ===

=== ITEM id=d0d04b5fdd7a80e4 ===
Gold sentences and facts:
Sentence 0:  Her favorite events are 800 metres and 1500 metres.
  Fact 0.0: This person's favorite events are the 800 metres.
  Fact 0.1: This person's favorite events are the 1500 metres.
Sentence 1:  The distance has been contested at the Summer Olympics since 1896 and the World Championships in Athletics since 1983.
  Fact 1.0: The distance has been contested at the Summer Olympics since 1896.
  Fact 1.1: The distance has been contested at the World Championships in Athletics since 1983.

Retrieved items:
<item 1>
Sinimole Paulose
Sinimol Paulose (born 24 June 1983) is an Indian middle distance runner.

 Her favorite events are 800 metres and 1500 metres.

 In 2009, she received Arjuna Award for her contribution to the Indian athletics.

 Sinimol was born in Piravom in Ernakulam district of Kerala state.

 She represented India at Beijing Olympics.
</item 1>

<item 2>
Athletics at the 2002 Asian Games
Athletics was contested from October 7 to October 14 at the 2002 Asian Games in Busan Asiad Main Stadium, Busan, South Korea.

 A total of 460 athletes (comprising 281 men and 179 women) from 39 nations took part in the competition.

 Afghanistan, Bhutan, Brunei, Laos and Maldives were the only nations without a representative in the events.
</item 2>

<item 3>
1500 metres
The 1500 metres or 1,500-metre run (typically pronounced 'fifteen-hundred metres') is the foremost middle distance track event in athletics.

 The distance has been contested at the Summer Olympics since 1896 and the World Championships in Athletics since 1983.

 It is equivalent to 1.5 kilometers or approximately ⁄  miles.
</item 3>

<item 4>
European 10,000m Cup
The European 10,000m Cup is an annual 10,000 metres race for European athletes which was first held in 1997.

 The competition is organised by the European Athletics Association and first began as the European 10000 Metres Challenge after the event was removed from the
European Cup programme.

 The competition has roots in the Iberian 10,000 metres Championships – a competition between Spanish and Portuguese athletes that was held between 1991 and
1996 – and the first five editions of the European 10000 Metres Challenge were held in the Iberian Peninsula.

 The event was first held under its current title in 2005.
</item 4>

<item 5>
Mary Chemweno
Mary Chemweno Koskei is a former Kenyan middle distance runner who specialised in 400 m, 800 m and 1500 m events.

 She won the gold medal in the inaugural 1979 African Championships in 800 metres, and another gold medal six years later in 1500 metres at the 1985 edition.

 She is also a three-time Kenyan national champion, having won both the 400 and 800 metres events in 1981 as well as the 800 m event again in 1986.

 Mid-career, in the early 1980s, she married long-distance runner Kipsubai Koskei.
</item 5>

<item 6>
Caster Semenya
Mokgadi Caster Semenya Bronze OIB (born 7 January 1991) is a South African middle-distance runner and 2016 Olympic gold medallist.

 Semenya won gold in the women's 800 metres at the 2009 World Championships with a time of 1:55.45 and at the 2017 World Championships in her new personal best,
1:55.16.

 Semenya also won silver medals at the 2011 World Championships and the 2012 Summer Olympics, both in the 800 metres.

 She was the winner of the gold medal in the 800 metre event at the 2016 Summer Olympics.
</item 6>

<item 7>
Siegmar Ohlemann
Siegmar ("Sig") Ohlemann (born May 11, 1938) is a retired middle distance runner, who represented Canada in three events (800 metres, 4 × 100 metres and 4 × 400
metres) at the 1960 Summer Olympics in Rome, Italy.

 He claimed the silver medal in 800 metres at the 1963 Pan American Games in Brazil, behind fellow Canadian Don Bertoia.
</item 7>

<item 8>
John Lindsay (Paralympian)
John Lindsay, OAM (born 29 January 1970) is an Australian Paralympic athlete from Melbourne.

 He competed in the 1988 Seoul games in distances ranging from 100 m to 800 m, but did not win any medals.

 At the 1992 Barcelona Games, he won a gold medal in the Men's 200 m TW3 event, for which he received a Medal of the Order of Australia, a silver medal in the
Men's 100 m TW3 event and a bronze medal in the Men's 400 m TW3 event.

 That year, he had a Victorian Institute of Sport scholarship.
</item 8>

<item 9>
John Lindsay (Paralympian)
That year, he had a Victorian Institute of Sport scholarship.

 He was also working as a fitness instructor in 1992, held world records in the 100 m and 200 m events, and was ranked 6th in the world in the 400 m.

 He won a gold medal in the men's athletics 100 m T52 event at the 1996 Summer Paralympics with a time of 15.22, a silver medal in the 200 m T52 event with a
time of 27.38, and a bronze medal in the 400 m T52 event with a time of 52.93.

 At the 2000 Sydney Games, he won a gold medal in the Men's 100 m T53 event, a silver medal as part of the Men's 4x100 m Relay T54 team, and a bronze medal in
the Men's 200 m T53 event; he was also part of the Men's 4x400 m Relay T54 team, which was the only one to qualify in its heat, but it did not make it to the
finals.
</item 9>

<item 10>
K. M. Binu
Kalayathumkuzhi Mathews Binu (born 20 December 1980) is an Indian track and field athlete from Kerala who specializes in 400 metres and 800 metres.

 He holds the current 400 metres national record of 45.48 s set at the 2004 Athens Olympics on 20 August 2004.

 He broke the 44-year-old Olympics mark (by an Indian) held by Milkha Singh who set an Indian National Record with a timing of 45.73 s at the 1960 Rome
Olympics.

 He and his elder sister K. M. Beenamol made history when they became the first Indian siblings to win medals in a major international competition.

 They won medals at the Busan Asian Games (2002).

 While Binu won the men's 800 metres silver, his sister won the gold medal in the women's event.

 Binu received the Arjuna Award for the year 2006 for his achievements in the Indian athletics.
</item 10>
=== END ITEM id=d0d04b5fdd7a80e4 ===

=== ITEM id=e0e9031a1ca03602 ===
Gold sentences and facts:
Sentence 0:  Related fields of study include Islamic philosophy and Sufism.
  Fact 0.0: Related fields of study include Islamic philosophy.
  Fact 0.1: Related fields of study include Sufism.
Sentence 1: Sufism or Taṣawwuf (Arabic: الْتَّصَوُّف [sect]; personal noun: صُوفِيّ - "ṣūfiyy"/ "ṣūfī", مُتَصَوّف - "mutaṣawwūf"), which is often defined as "Islamic mysticism", "the inward dimension of Islam", or "the phenomenon of mysticism within Islam", is a mystical trend in Islam "characterized ... [by particular] values, ritual practices, doctrines and institutions" which began very early on in Islamic history and which represents "the main manifestation and the most important and central crystallization of" mystical practice in Islam.
  Fact 1.0: Sufism is also called Taṣawwuf.
  Fact 1.1: The Arabic name for Sufism is الْتَّصَوُّف [sect].
  Fact 1.2: The personal noun for a Sufi is صُوفِيّ, transliterated as 'ṣūfiyy' or 'ṣūfī'.
  Fact 1.3: The personal noun mutaṣawwūf is مُتَصَوّف.
  Fact 1.4: Sufism is often defined as 'Islamic mysticism'.
  Fact 1.5: Sufism is often defined as 'the inward dimension of Islam'.
  Fact 1.6: Sufism is often defined as 'the phenomenon of mysticism within Islam'.
  Fact 1.7: Sufism is a mystical trend in Islam.
  Fact 1.8: Sufism is characterized by particular values, ritual practices, doctrines and institutions.
  Fact 1.9: Sufism began very early on in Islamic history.
  Fact 1.10: Sufism represents the main manifestation of mystical practice in Islam.
  Fact 1.11: Sufism represents the most important and central crystallization of mystical practice in Islam.

Retrieved items:
<item 1>
Madonna Studies
Madonna Studies is a term which has been used to refer to a development of a field in several studios since late 1980s.

 One writer described the "institutionalization of a major subdivision of American media studies into Madonna studies".

 The field related to the study of and interpretation of the work of American pop musician Madonna using an interdisciplinary approach incorporating cultural
studies and media studies.

 A notable compendium of essays titled "The Madonna Connection" was published in 1992.

 Controversy over this field of study stemmed from discussions over the intellectual worth of pursuing academic inquiry into a pop musician, with some arguing
the field was nothing more than pop cultural commentary.
</item 1>

<item 2>
A Hidden Treasure
Hidden Treasure (Arabic: کنزاً مخفیاً‎ ‎ ) is a Hadith Qudsi that has a very prominent role in Islamic mysticism and Islamic philosophy.
</item 2>

<item 3>
Sufism
Sufism or Taṣawwuf (Arabic: الْتَّصَوُّف [sect]; personal noun: صُوفِيّ - "ṣūfiyy"/ "ṣūfī", مُتَصَوّف - "mutaṣawwūf"), which is often defined as "Islamic
mysticism", "the inward dimension of Islam", or "the phenomenon of mysticism within Islam", is a mystical trend in Islam "characterized ... [by particular]
values, ritual practices, doctrines and institutions" which began very early on in Islamic history and which represents "the main manifestation and the most
important and central crystallization of" mystical practice in Islam.

 Although the overwhelming majority of Sufis, both pre-modern and modern, have been adherents of Sunni Islam, there nevertheless also developed certain strands
of Sufi practice within the ambit of Shia Islam during the late medieval period.
</item 3>

<item 4>
Ovamir Anjum
Ovamir Anjum is the Imam Khattab Chair of Islamic Studies at the Department of Philosophy, University of Toledo.

 He studies the connections between theology, ethics, politics, and law in classical and medieval Islam, with a subfocus on its comparisons with western
thought.

 Related fields of study include Islamic philosophy and Sufism.
</item 4>

<item 5>
Omid Safi
Omid Safi is an American Professor of Asian and Middle Eastern Studies at Duke University, where he is the Director of Duke Islamic Studies Center, and a
columnist for On Being.

 Dr. Safi specializes in Islamic mysticism (Sufism), contemporary Islamic thought and medieval Islamic history.

 He has served on the board of the Pluralism project at Harvard University and is the co-chair of the steering committee for the Study of Islam and the Islamic
Mysticism Group at the American Academy of Religion.

 Before joining Duke University, Dr. Safi was a professor at the University of North Carolina Chapel Hill.
</item 5>

<item 6>
Moinuddin Chishti
1325) and Amir Khusrow (d. 1325).

 As such, Muʿīn al-Dīn Chishtī's legacy rests primarily on his having been "one of the most outstanding figures in the annals of Islamic mysticism."

 Additionally, Muʿīn al-Dīn Chishtī is also notable for having been one of the first major Islamic mystics to formally allow his followers to incorporate the
"use of music" in their devotions, liturgies, and hymns to God, which he did so in order to make the foreign Arab faith more relatable to the indigenous peoples
who had recently entered the religion or whom he sought to convert.
</item 6>

<item 7>
Animality studies
Animality studies is an emerging interdisciplinary academic field focused on the cultural study of animals and animality.

 It can be distinguished from animal studies and critical animal studies by its resistance to animal rights or animal welfare as an explicit justification for
work in this field.

 According to Michael Lundblad, "If animal studies can be seen as work that explores representations of animality and related discourses with an emphasis on
advocacy for nonhuman animals, animality studies becomes work that emphasizes the history of animality in relation to human cultural studies, without an
explicit call for nonhuman advocacy."
</item 7>

<item 8>
Irfan
In Islam, ‘Irfaan (Arabic/Persian/Urdu: عرفان ; Turkish: "İrfan" ), also spelt "Irfaan" and "Erfan", literally ‘knowledge, awareness, wisdom’, is gnosis.

 Islamic mysticism can be considered as a vast range that engulfs theoretical and practical and conventional mysticism and has been intertwined with sufism and
in some cases they are assumed identical.

 however Islamic mysticism is assumed as one of the Islamic sciences alongside theology and philosophy.

 Islamic’s mysticism is cognition and knowledge that love has been intertwined through it with structure of revelation in Islam.
</item 8>

<item 9>
Dawud Tai
Abu Solaiman Dawud ibn Nosair al-Ta’i, usually referred to as Dawud Tai, (died between 777 and 782) was a Islamic scholar and Sufi mystic.

 His disciples included many influential personalities of Islamic mysticism, e.g., Maruf Karkhi and Ahmad al-Buni.
</item 9>

<item 10>
Reynold A. Nicholson
Reynold Alleyne Nicholson, or R. A. Nicholson (18 August 1868 – 27 August 1945), was an eminent English orientalist, scholar of both Islamic literature and
Islamic mysticism and widely regarded as one of the greatest Rumi (Mevlana or Mawlana) scholars and translators in the English language.
</item 10>
=== END ITEM id=e0e9031a1ca03602 ===

=== ITEM id=38094abf58405da7 ===
Gold sentences and facts:
Sentence 0:  The breed is used for hunting in packs and descends from the old Hound of Saintonge type of large hunting dog.
  Fact 0.0: The breed is used for hunting in packs.
  Fact 0.1: The breed descends from the old Hound of Saintonge type of large hunting dog.
Sentence 1: The Kritikos Lagonikos (Cretan Hound) (Greek:Kρητικός Λαγωνικός) is a hunting breed of dog from the island of Crete, in Greece.
  Fact 1.0: The Kritikos Lagonikos, also called the Cretan Hound, is a hunting breed of dog.
  Fact 1.1: The Kritikos Lagonikos has the Greek name Κρητικός Λαγωνικός.
  Fact 1.2: The Kritikos Lagonikos is from the island of Crete, in Greece.

Retrieved items:
<item 1>
Ariegeois
The Ariegeois is a breed of dog from the "département" of Ariège in the Midi-Pyrenées region of southern France.

 It is a medium-sized pack-hunting scenthound deriving from crossing of Grand Bleu de Gascogne and Grand Gascon-Saintongeois hounds with local Briquet dogs.

 It is used both as a courser and for driving game to waiting guns.

 While most successful with hares, it is also used for hunting deer and boar.

 It is distinguished by its friendly nature with other hounds and affection for human companions.
</item 1>

<item 2>
Grand Anglo-Français Blanc et Noir
The Grand Anglo-Français Blanc et Noir (FCI No.323) translated into English as the Great Anglo-French White and Black Hound, is a breed of dog used in hunting
as a scenthound, usually in packs.

 It is one of the Anglo-French hound breeds which were created by crossing French scenthounds with English (Anglo) Foxhounds.
</item 2>

<item 3>
Braque Francais
The Braques français are hunting dogs, from a very old type of gun dog used for pointing the location of game birds for a hunter.

 There are two breeds of Braque français, both from the south of France, the Braque français, type Gascogne (French Pointing Dog - Gascogne type, larger size)
and the Braque français, type Pyrénées (French Pointing Dog - Pyrenean type, smaller size) They are popular hunting dogs in France, but are seldom seen
elsewhere.
</item 3>

<item 4>
Chien Français Tricolore
The Chien Français Tricolore (FCI No.219) translated into English as the French Tricolour Hound, is a breed of dog of the scenthound type, originating in
France.

 The breed is used for hunting in packs.
</item 4>

<item 5>
Grand Bleu de Gascogne
The Grand Bleu de Gascogne (FCI No.22) is a breed of dog of the scenthound type, originating in France and used for hunting in packs.

 Today's breed is the descendant of a very old type of large hunting dog, and is an important breed in the ancestry of many other hounds.
</item 5>

<item 6>
Griffon Bleu de Gascogne
The Griffon Bleu de Gascogne (FCI No.32) Is a breed of dog of the scenthound type, originating in France, and is a versatile hunting dog, used on small and
large game, in packs or individually.

 The Griffon Bleu de Gascogne has a speckled, rough coat.
</item 6>

<item 7>
Grand Anglo-Français Blanc et Orange
The Grand Anglo-Français Blanc et Orange (FCI No.324) translated into English as the Great Anglo-French White and Orange Hound, is a breed of dog used in
hunting as a scenthound, usually in packs.

 It is one of the Anglo-French hound breeds which were created by crossing French scenthounds with English (Anglo) foxhounds.
</item 7>

<item 8>
Petit Bleu de Gascogne
The Petit Bleu de Gascogne (FCI No.31) Is a breed of dog of the scenthound type, originating in France and used for hunting in packs.

 Today's breed is the descendant of a very old type of large hunting dog.

 The Petit Bleu de Gascogne is not a small (petite) dog, the name comes from its use on small game.
</item 8>

<item 9>
Chien Français Blanc et Noir
The Chien Français Blanc et Noir (FCI No.220) translated into English as the French White and Black Hound, is a breed of dog of the scenthound type, originating
in France.

 The breed is used for hunting in packs and descends from the old Hound of Saintonge type of large hunting dog.
</item 9>

<item 10>
Chien Français Blanc et Orange
The Chien Français Blanc et Orange (FCI No.316) is a breed of dog of the scenthound type, originating in France.

 The breed is used for hunting in packs and descends from the old Hound of Saintonge type of large hunting dog.
</item 10>
=== END ITEM id=38094abf58405da7 ===

=== ITEM id=d93c8c1a55d61657 ===
Gold sentences and facts:
Sentence 0: Martin Degville is (born 27 January 1961 in Walsall) is the lead singer and co-songwriter of the UK pop band, Sigue Sigue Sputnik – which had a worldwide hit single in 1986 with "Love Missile F1-11" – and six other EMI single releases.
  Fact 0.0: Martin Degville was born 27 January 1961 in Walsall.
  Fact 0.1: Martin Degville is the lead singer of the UK pop band Sigue Sigue Sputnik.
  Fact 0.2: Martin Degville is a co-songwriter of Sigue Sigue Sputnik.
  Fact 0.3: Sigue Sigue Sputnik had a worldwide hit single in 1986 with 'Love Missile F1-11'.
  Fact 0.4: Sigue Sigue Sputnik has six other EMI single releases.
Sentence 1: Paul Richard "Richie" Furay (born May 9, 1944) is an American singer, songwriter, and Rock & Roll Hall of Fame member who is best known for forming the bands Buffalo Springfield with Stephen Stills, Neil Young, Bruce Palmer, and Dewey Martin, and Poco with Jim Messina, Rusty Young, George Grantham and Randy Meisner.
  Fact 1.0: Paul Richard 'Richie' Furay was born May 9, 1944.
  Fact 1.1: Richie Furay is an American singer.
  Fact 1.2: Richie Furay is an American songwriter.
  Fact 1.3: Richie Furay is a Rock & Roll Hall of Fame member.
  Fact 1.4: Richie Furay is best known for forming Buffalo Springfield with Stephen Stills, Neil Young, Bruce Palmer, and Dewey Martin.
  Fact 1.5: Richie Furay is best known for forming Poco with Jim Messina, Rusty Young, George Grantham, and Randy Meisner.

Retrieved items:
<item 1>
Poco: The Songs of Richie Furay
Poco: The Songs of Richie Furay is a compilation album consisting of songs by Richie Furay during his tenure with the band Poco, released in 1980.
</item 1>

<item 2>
Jerry Garcia
Jerome John Garcia (August 1, 1942 – August 9, 1995) was an American singer-songwriter and guitarist, best known for his work as the lead guitarist and as a
vocalist with the band the Grateful Dead, which came to prominence during the counterculture era in the 1960s.

 Although he disavowed the role, Garcia was viewed by many as the leader or "spokesman" of the group.
</item 2>

<item 3>
Martin Degville
Martin Degville is (born 27 January 1961 in Walsall) is the lead singer and co-songwriter of the UK pop band, Sigue Sigue Sputnik – which had a worldwide hit
single in 1986 with "Love Missile F1-11" – and six other EMI single releases.

 Sputnik was formed with ex-Generation X bassist Tony James.
</item 3>

<item 4>
Sunshine Becker
Sunshine Becker (born Sunshine Flower Garcia on July 1, 1972) is an American singer who performed backing vocals for the band Furthur.

 Despite her maiden name, Garcia, she is not related to Jerry Garcia, an incorrect assumption made by some because of her involvement with Furthur, a
post-Garcia incarnation of the Grateful Dead.

 Similarly, despite her first name, Sunshine, she is not to be confused with Sunshine Kesey, daughter of Ken Kesey and Carolyn Adams (aka Mountain Girl or MG),
Jerry Garcia's second wife.
</item 4>

<item 5>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 5>

<item 6>
Dean Martin
Dean Martin (born Dino Paul Crocetti; June 7, 1917 – December 25, 1995) was an American singer, actor, comedian, and film producer.

 One of the most popular and enduring American entertainers of the mid-20th century, Martin was nicknamed the "King of Cool" for his seemingly effortless
charisma and self-assurance.
</item 6>

<item 7>
Richie Furay
Paul Richard "Richie" Furay (born May 9, 1944) is an American singer, songwriter, and Rock & Roll Hall of Fame member who is best known for forming the bands
Buffalo Springfield with Stephen Stills, Neil Young, Bruce Palmer, and Dewey Martin, and Poco with Jim Messina, Rusty Young, George Grantham and Randy Meisner.

 His best known song (originally written during his tenure in Buffalo Springfield, but eventually performed by Poco, as well) was "Kind Woman", which he wrote
for his wife, Nancy.
</item 7>

<item 8>
Dean Martin Sings
Dean Martin Sings is the first studio album by Italian-American singer Dean Martin, released in 1953.

 It is the first long-play 10-inch album recorded by Martin for Capitol Records during two sessions recorded on the evening of November 20, 1952.

 The first session was recorded between 5 and 8 PM and it produced five songs featuring string arrangements.

 "There's My Lover" was recorded but not released.

 After a ninety-minute break, Martin was joined by a brass arrangement to record the remaining four songs.

 Seven of the eight songs on this album appeared in the Martin & Lewis film, "The Stooge".

 Two years later, the songs from this 10-inch album would be combined with four newly popular songs recorded between 1951 and 1953 to create a full-length
12-inch album.

 The 2005 Collectors' Choice reissue added four bonus songs recorded between 1949 and 1953 and was released with alternative cover artwork.
</item 8>

<item 9>
Souther–Hillman–Furay Band
The Souther Hillman Furay Band (SHF) was a country rock supergroup led by singer-songwriters Richie Furay (Buffalo Springfield, Poco), Chris Hillman (The Byrds,
The Flying Burrito Brothers, Manassas), and J.D. Souther (Longbranch Pennywhistle, noted songwriter for Linda Ronstadt and the Eagles).

 The band recorded two albums during the 1970s before breaking up due to disagreements between the members.
</item 9>

<item 10>
Martin Flavin
Martin Archer Flavin (November 2, 1883 – December 27, 1967) was an American playwright and novelist.
</item 10>
=== END ITEM id=d93c8c1a55d61657 ===

=== ITEM id=f7fd7f56455aa1d2 ===
Gold sentences and facts:
Sentence 0:  He was the brother of Alfred Jodl, Chief of the Operations Staff of the OKW.
  Fact 0.0: The person described in this sentence was the brother of Alfred Jodl.
  Fact 0.1: Alfred Jodl was Chief of the Operations Staff of the OKW.
Sentence 1: Alfred Josef Ferdinand Jodl (   ; 10 May 1890 – 16 October 1946) was a German general and war criminal during World War II, who served as the Chief of the Operations Staff of the Armed Forces High Command (Oberkommando der Wehrmacht).
  Fact 1.0: Alfred Josef Ferdinand Jodl lived from 10 May 1890 to 16 October 1946.
  Fact 1.1: Alfred Jodl was a German general.
  Fact 1.2: Alfred Jodl was a war criminal during World War II.
  Fact 1.3: Alfred Jodl served as Chief of the Operations Staff of the Armed Forces High Command (Oberkommando der Wehrmacht).

Retrieved items:
<item 1>
Hasso von Wedel (general)
Hasso von Wedel (20 November 1898 – 3 January 1961) was a German general who commanded the Wehrmacht Propaganda Troops during World War II.

 He was directly subordinate to the head of OKW Operations Staff ("Wehrmachtführungsstab, WFSt.")

, General Alfred Jodl.

 Wedel's Propaganda Department had control over the propaganda units and served to mediate between them and the Reich Propaganda Ministry of Joseph Goebbels.
</item 1>

<item 2>
List of Generalobersts
Generaloberst, in English Colonel General, was, in Germany and Austria-Hungary—the German "Reichswehr" and "Wehrmacht", the Austro-Hungarian Common Army, and
the East German National People's Army, as well as the respective police services—the second highest general officer rank, ranking above full general but below
general field marshal.

 It was equivalent to "Generaladmiral" in the "Kriegsmarine" until 1945, or to Flottenadmiral in the "Volksmarine" until 1990.

 The rank was the highest ordinary military rank and the highest military rank awarded in peacetime; the higher rank of general field marshal was only awarded
in wartime by the head of state.

 In general, a Generaloberst had the same privileges as a general field marshal.
</item 2>

<item 3>
Alfred Jodl
Alfred Josef Ferdinand Jodl (   ; 10 May 1890 – 16 October 1946) was a German general and war criminal during World War II, who served as the Chief of the
Operations Staff of the Armed Forces High Command (Oberkommando der Wehrmacht).
</item 3>

<item 4>
Walter Dornberger
Major-General Dr. Walter Robert Dornberger (6 September 1895 – 27 June 1980) was a German Army artillery officer whose career spanned World War I and World War
II.

 He was a leader of Nazi Germany's V-2 rocket program and other projects at the Peenemünde Army Research Center.
</item 4>

<item 5>
Ludwig Beck
Ludwig August Theodor Beck (29 June 1880 – 21 July 1944) was a German general and Chief of the German General Staff during the early years of the Nazi regime in
Germany before World War II.

 Ludwig Beck was never a member of the Nazi Party, though in the early 1930s he supported Adolf Hitler's forceful denunciation of the Versailles Treaty and
belief in the need for Germany to rearm.

 Beck had grave misgivings regarding the Nazi demand that all German officers swear an oath of fealty to the person of Hitler in 1934, though he believed that
Germany needed strong government and that Hitler could successfully provide this so long as he was influenced by traditional elements within the military rather
than the SA and SS.
</item 5>

<item 6>
Helmuth von Moltke the Younger
Helmuth Johann Ludwig von Moltke (] ; 23 May 1848 – 18 June 1916), also known as Moltke the Younger, was a nephew of "Generalfeldmarschall" (Field Marshal)
Helmuth Karl Bernhard von Moltke and served as the Chief of the German General Staff from 1906 to 1914.

 The two are often differentiated as "Moltke the Elder" and "Moltke the Younger".
</item 6>

<item 7>
Friedrich Jeckeln
Friedrich Jeckeln (2 February 1895 – 3 February 1946) was a high-ranking official in the SS of Nazi Germany who served as an SS and Police Leader in the
occupied Soviet Union during World War II.

 Jeckeln was the commander of one of the largest collection of "Einsatzgruppen" and was personally responsible for ordering and organizing the deaths of over
100,000 Jews, Slavs, Romani, and other "undesirables".

 After the end of World War II, Jeckeln was convicted for his crimes by a Soviet military tribunal in Riga, Latvia and executed in 1946.
</item 7>

<item 8>
Ferdinand Jodl
Ferdinand Alfred Friedrich Jodl (28 November 1896 – 9 June 1956) was a German general during World War II who commanded the Mountain Corps Norway during the
Petsamo–Kirkenes Offensive.

 He was the brother of Alfred Jodl, Chief of the Operations Staff of the OKW.

 Ferdinand Jodl was acquitted by a German court.
</item 8>

<item 9>
Franz Halder
Franz Halder (30 June 1884 – 2 April 1972) was a German general and the chief of the "Oberkommando des Heeres" staff (OKH, Army High Command) from 1938 until
September 1942, when he was dismissed after frequent disagreements with Adolf Hitler.

 Until December 1941 Halder's military position corresponded to the old Chief of the General Staff position, which during World War I had been the highest
military office in the German Imperial Army.

 Halder's diary during his time as chief of OKH General Staff has been a source for authors that have written about such subjects as Hitler, World War II and
the Nazi Party.

 In William Shirer's "The Rise and Fall of the Third Reich", Halder's diary is cited hundreds of times.
</item 9>

<item 10>
Herbert Büchs
Herbert Büchs (20 November 1913 – 19 May 1996) was a Lieutenant General of the German Air Force and a former Luftwaffe staff officer in Nazi Germany's
Oberkommando der Wehrmacht (OKW) during World War II.

 As second adjutant to General Alfred Jodl with the rank of Major he is notable for being present in the conference room when the 20 July plot bomb exploded in
1944.
</item 10>
=== END ITEM id=f7fd7f56455aa1d2 ===

=== ITEM id=f93b894826970184 ===
Gold sentences and facts:
Sentence 0: Wilmington International Airport (IATA: ILM, ICAO: KILM, FAA LID: ILM) is a public airport located just north of Wilmington, North Carolina, in unincorporated Wrightsboro, Cape Fear Township, New Hanover County.
  Fact 0.0: Wilmington International Airport has the IATA code ILM.
  Fact 0.1: Wilmington International Airport has the ICAO code KILM.
  Fact 0.2: Wilmington International Airport has the FAA LID ILM.
  Fact 0.3: Wilmington International Airport is a public airport.
  Fact 0.4: Wilmington International Airport is located just north of Wilmington, North Carolina.
  Fact 0.5: Wilmington International Airport is located in unincorporated Wrightsboro, Cape Fear Township, New Hanover County.
Sentence 1: Texarkana Regional Airport (IATA: TXK, ICAO: KTXK, FAA LID: TXK) , also known as Webb Field, is a public use airport located three nautical miles (6 km) northeast of the central business district of Texarkana, a city in Miller County, Arkansas, United States.
  Fact 1.0: Texarkana Regional Airport has the IATA code TXK.
  Fact 1.1: Texarkana Regional Airport has the ICAO code KTXK.
  Fact 1.2: Texarkana Regional Airport has the FAA LID TXK.
  Fact 1.3: Texarkana Regional Airport is also known as Webb Field.
  Fact 1.4: Texarkana Regional Airport is a public use airport.
  Fact 1.5: Texarkana Regional Airport is located three nautical miles (6 km) northeast of the central business district of Texarkana.
  Fact 1.6: Texarkana is a city in Miller County, Arkansas, United States.

Retrieved items:
<item 1>
Wilmington Airport (Delaware)
Wilmington Airport (IATA: ILG) (also known as Wilmington/Philadelphia Regional Airport, New Castle County Airport, or New Castle Airport) is an airport located
in unincorporated New Castle County, Delaware near Wilmington, Delaware.

 Owned by the Delaware River and Bay Authority, it is five miles (8 km) south of Wilmington and about 30 miles (50 km) from Philadelphia.

 It is included in the Federal Aviation Administration (FAA) National Plan of Integrated Airport Systems for 2017–2021, in which it is categorized as a non-hub
primary commercial service facility.
</item 1>

<item 2>
Sharp County Regional Airport
Sharp County Regional Airport (IATA: CKK, ICAO: KCVK, FAA LID: CVK) is a public-use airport in Sharp County, Arkansas, United States.

 It is located three nautical miles (4 mi, 6 km) northeast of the central business district of Ash Flat, Arkansas.

 The airport is owned by the Sharp County Regional Airport Authority.
</item 2>

<item 3>
Texarkana Regional Airport
Texarkana Regional Airport (IATA: TXK, ICAO: KTXK, FAA LID: TXK) , also known as Webb Field, is a public use airport located three nautical miles (6 km)
northeast of the central business district of Texarkana, a city in Miller County, Arkansas, United States.

 It is owned by the Texarkana Airport Authority.

 The airport is located within the city limits of Texarkana, roughly 3 miles east of State Line Avenue.

 The front gate opens to the northwest, at the intersection of Arkansas Boulevard and U.S. Route 67.

 A Union Pacific Railroad line runs parallel to US 67 on the side of the highway facing the airport.

 It is mostly used for general aviation, but is also served by American Eagle.
</item 3>

<item 4>
Wilmington International Airport
Wilmington International Airport (IATA: ILM, ICAO: KILM, FAA LID: ILM) is a public airport located just north of Wilmington, North Carolina, in unincorporated
Wrightsboro, Cape Fear Township, New Hanover County.
</item 4>

<item 5>
Air Wisconsin
Air Wisconsin Airlines is a regional airline based at Appleton International Airport in the town of Greenville, Wisconsin, United States, near Appleton.

 Air Wisconsin is the largest privately held regional airline in the United States.

 It currently operates regional jet flights as American Eagle under contract to American Airlines via a code sharing agreement, serving cities in the U.S. and
Canada with hubs at Boston Logan International Airport (BOS), Charlotte Douglas International Airport (CLT), Philadelphia International Airport (PHL) and Ronald
Reagan Washington National Airport (DCA) .

 Air Wisconsin previously operated United Express service on behalf of United Airlines followed by US Airways Express service on behalf of US Airways prior to
becoming an American Eagle regional air carrier.

 The company has announced it will once again partner with United Airlines as a United Express code sharing air carrier with primary hubs to be located at
Chicago O'Hare International Airport (ORD) and Washington Dulles International Airport (IAD) beginning in September 2017 and will be "exclusively operating" as
United Express by March 2018.
</item 5>

<item 6>
Atlantic Southeast Airlines Flight 2311
Atlantic Southeast Airlines Flight 2311 was a regularly scheduled commuter flight from Hartsfield–Jackson Atlanta International Airport to Glynco Jetport (since
renamed Brunswick Golden Isles Airport) in Brunswick, Georgia on April 5, 1991.

 The flight, operated using a twin-turboprop Embraer EMB 120 Brasilia, crashed just north of Brunswick while approaching the airport for landing.

 All 23 people aboard the plane were killed, including passengers Sonny Carter and John Tower.

 Four years later, another Embraer Brasilia of ASA crashed in the Georgia countryside in similar circumstances, with nine fatalities.
</item 6>

<item 7>
Raleigh Executive Jetport
Raleigh Exec: The Raleigh Executive Jetport @ Sanford-Lee County or Raleigh Exec Jetport at Sanford-Lee CountyFAA Airport Master Record for TTA (Form 5010 )
(ICAO: KTTA, FAA LID: TTA) is a public use airport located seven nautical miles (8 mi, 13 km) northeast of the central business district of Sanford, a city in
Lee County, North Carolina, United States.

 It is owned by the Sanford-Lee County Regional Airport Authority and was previously known as Sanford-Lee County Regional Airport.

 This airport is included in the National Plan of Integrated Airport Systems for 2011–2015, which categorized it as a "reliever airport" for Raleigh-Durham
International Airport.
</item 7>

<item 8>
Pensacola International Airport
Pensacola International Airport (IATA: PNS, ICAO: KPNS, FAA LID: PNS) , formerly Pensacola Gulf Coast Regional Airport and Pensacola Regional Airport (Hagler
Field), is a public use airport three nautical miles (6 km) northeast of the central business district of Pensacola, in Escambia County, Florida, United States.

 It is owned by the City of Pensacola.

 Despite the name, this airport does not offer direct international flights.

 This airport is one of the five major airports in North Florida, others being: Northwest Florida Beaches International Airport, Destin-Fort Walton Beach
Airport Tallahassee International Airport, and Jacksonville International Airport.
</item 8>

<item 9>
San Antonio International Airport
San Antonio International Airport (IATA: SAT, ICAO: KSAT, FAA LID: SAT) is an international airport located in San Antonio, Texas and serving the Greater San
Antonio metropolitan area.

 It is located in Uptown Central San Antonio, about 8 miles north of Downtown.

 Its has three runways and covers 2,305 acres (933 ha).

 Its elevation is 809 ft above sea level.

 SAT is a Class C airport.
</item 9>

<item 10>
Dane County Regional Airport
Dane County Regional Airport (DCRA) (IATA: MSN, ICAO: KMSN, FAA LID: MSN) (Truax Field) is a civil-military airport located six miles northeast of downtown
Madison, the capital of Wisconsin.

 In the Federal Aviation Administration (FAA) National Plan of Integrated Airport Systems for 2017–2021, it is categorized as a small-hub primary commercial
service facility.
</item 10>
=== END ITEM id=f93b894826970184 ===
