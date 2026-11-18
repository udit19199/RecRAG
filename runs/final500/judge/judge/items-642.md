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

This file has 10 items.

=== ITEM id=6e7be7de8e765e0d ===
Gold sentences and facts:
Sentence 0: Like many of Marivaux's , "La Surprise de l'amour" makes use of stock characters from the Commedia dell'arte.
  Fact 0.0: Like many of Marivaux's plays, La Surprise de l'amour makes use of stock characters from the Commedia dell'arte.
  Fact 0.1: Many of Marivaux's plays make use of stock characters from the Commedia dell'arte.
Sentence 1:  In this play, Arlequin and Columbine are featured.
  Fact 1.0: In La Surprise de l'amour, Arlequin is featured.
  Fact 1.1: In La Surprise de l'amour, Columbine is featured.
Sentence 2: Harlequin ( ; Italian: "Arlecchino" ] , French: "Arlequin" ] , Old French "Harlequin") is the best-known of the "zanni" or comic servant characters from the Italian "Commedia dell'arte".
  Fact 2.0: Harlequin is the best-known of the zanni, the comic servant characters from the Italian Commedia dell'arte.
  Fact 2.1: Harlequin is called Arlecchino in Italian.
  Fact 2.2: Harlequin is called Arlequin in French.
  Fact 2.3: Harlequin is called Harlequin in Old French.

Retrieved items:
<item 1>
Columbina
Columbina (in Italian Colombina, meaning "little dove"; in French and English Colombine) is a stock character in the "Commedia dell'Arte".

 She is Harlequin's mistress, a comic servant playing the tricky slave type, and wife of Pierrot.

 Rudlin and Crick use the Italian spelling Colombina in "Commedia dell'arte: A Handbook for Troupes".
</item 1>

<item 2>
Tartaglia (commedia dell'arte)
Tartaglia is a dainty character in the Commedia dell'arte.

 He is farsighted and with a minor stutter (hence his name; cf. Spanish tartamudear), he is usually classed as one of the group of old characters (vecchio) who
appears in many scenarios as one of the lovers (innamorati).

 His social status varies; he is sometimes a bailiff, lawyer, notary or chemist.

 Dramatist Carlo Gozzi turned him into a statesman, and so he remained thereafter.

 Tartaglia wears a large felt hat, an enormous cloak, oversized boots, a long sword, a giant moustache and a cardboard nose.

 He usually represents the lower working class but at times the middle or upper class in the commedia dell'arte.

 In the opera "Le maschere" by Mascagni, one of the characters is Tartaglia, a stuttering servant.
</item 2>

<item 3>
Commedia dell'arte
Commedia dell'arte (] , comedy of the profession) was an early form of professional theatre, originating from Italy, that was popular in Europe from the 16th
through the 18th century.

 "Commedia dell'arte" also is known as "commedia alla maschera", "commedia improvviso", and "commedia dell'arte all'improvviso".

 "Commedia" is a form of theatre characterized by masked "types" which began in Italy in the 16th century and was responsible for the advent of actresses
(Isabella Andreini) and improvised performances based on sketches or scenarios.

 A "commedia", such as "The Tooth Puller", is both scripted and improvised.

 Characters' entrances and exits are scripted.

 A special characteristic of "commedia dell'arte" are the "lazzi".

 A "lazzo" is a joke or "something foolish" or "witty".

 Another characteristic of "commedia dell'arte" is pantomime, which is mostly used by the character Arlecchino (Harlequin).
</item 3>

<item 4>
Pedrolino
Pedrolino is a "primo zanni", or comic servant, of the "Commedia dell'Arte"; the name is a hypocorism of "Pedro" (Peter), via the suffix "-lino".

 The character made its first appearance in the last quarter of the 16th century, apparently as the invention of the actor with whom the role was to be long
identified, Giovanni Pellesini.

 Contemporary illustrations suggest that his white blouse and trousers constituted "a variant of the typical "zanni" suit", and his Bergamasque dialect marked
him as a member of the "low" rustic class.

 But if his costume and social station were without distinction, his dramatic role was certainly not: as a multifaceted "first" "zanni", his character was—and
still is—rich in comic incongruities.
</item 4>

<item 5>
La Surprise de l'amour
Like many of Marivaux's , "La Surprise de l'amour" makes use of stock characters from the Commedia dell'arte.

 In this play, Arlequin and Columbine are featured.
</item 5>

<item 6>
Gilles (stock character)
Gilles (] )—sometimes Gille—is a stock character of French farce and Commedia dell'Arte.

 He enjoyed his greatest vogue in 18th-century France, in entertainments both at the fairgrounds of the capital and in private and public theaters, though his
origins can be traced back to the 17th century and, possibly, the century previous.

 A "zanni", or comic servant, he is a type of bungling clown, stupid, credulous, and lewd—a character that shares little, problematically, with the sensitive
figure in Watteau's famous portrait that, until the latter half of the 20th century, bore his name alone.

 Gilles fades from view in the 19th century, to persist in the 20th and 21st as the Belgian Gilles of Binche Carnival.
</item 6>

<item 7>
Commedia dell’arte, Italy’s 16th–18th-century professional masked theatre, combined scripted scenarios, improvisation, pantomime and comic lazzi. Stock figures
included Harlequin, Columbina, Pierrot, Pedrolino, Tartaglia and French clown-servant Gilles. Niccolò Barbieri (1586–1641), actor-writer known as Beltrame,
created Scapino; his play *L’inavertito* inspired M
</item 7>

<item 8>
Harlequin
Harlequin ( ; Italian: "Arlecchino" ] , French: "Arlequin" ] , Old French "Harlequin") is the best-known of the "zanni" or comic servant characters from the
Italian "Commedia dell'arte".

 The role is traditionally believed to have been introduced by Zan Ganassa in the late 16th century, and was definitively popularized by the Italian actor
Tristano Martinelli in Paris in 1584–1585 and became a stock character after Martinelli's death in 1630.
</item 8>

<item 9>
Niccolò Barbieri
Niccolò (or Nicolò) Barbieri (Vercelli, 1586 - 1641) was an Italian writer and actor of the commedia dell'arte theatrical genre.

 He was also known as Beltrame di Milano ("Beltrame of Milan") in reference to one of his most popular characters, Beltrame; this was the main character of one
of Barbieri's best known plays, "L'inavertito", which is known to have inspired Moliere's "L'Étourdi ou Les Contretemps" ("The Blunderer").

 Besides popularizing Beltrame, Barbieri is also credited with creating another "commedia dell'arte" mask, Scapino.
</item 9>

<item 10>
Bernhard Bötel
His stage repertoire included Belmonte in Wolfgang Amadeus Mozart's "Die Entführung aus dem Serail", Chapelou in Adolphe Adam's "Le postillon de Lonjumeau",
Count Almaviva in Gioachino Rossini's "The Barber of Seville", Daniel in Franz Lehár's "The Merry Widow", the Duke of Mantua in Giuseppe Verdi's "Rigoletto",
Gabriel von Eisenstein in Johann Strauss II's "Die Fledermaus", Indigo in Strauss' "Indigo und die vierzig Räuber", Jeník in Bedřich Smetana's "The Bartered
Bride", Paolino in Domenico Cimarosa's "Il matrimonio segreto", Pâris in Jacques Offenbach's "La belle Hélène", Pietro in Franz von Suppé's "Boccaccio", and
Wilhelm Meister in Ambroise Thomas' "Mignon".
</item 10>
=== END ITEM id=6e7be7de8e765e0d ===

=== ITEM id=dc347b5f22198567 ===
Gold sentences and facts:
Sentence 0: Icehouse pieces, or Icehouse Pyramids, Treehouse pieces, Treehouse Pyramids and officially Looney Pyramids, are nestable and stackable pyramid-shaped gaming pieces and a game system.
  Fact 0.0: Icehouse pieces are also called Icehouse Pyramids.
  Fact 0.1: Treehouse pieces are also called Treehouse Pyramids.
  Fact 0.2: Icehouse pieces, Icehouse Pyramids, Treehouse pieces, and Treehouse Pyramids are officially called Looney Pyramids.
  Fact 0.3: Icehouse pieces, Icehouse Pyramids, Treehouse pieces, Treehouse Pyramids, and Looney Pyramids are nestable pyramid-shaped gaming pieces.
  Fact 0.4: Icehouse pieces, Icehouse Pyramids, Treehouse pieces, Treehouse Pyramids, and Looney Pyramids are stackable pyramid-shaped gaming pieces.
  Fact 0.5: Icehouse pieces, Icehouse Pyramids, Treehouse pieces, Treehouse Pyramids, and Looney Pyramids are a game system.
Sentence 1:  The game system was invented by Andrew Looney and John Cooper in 1987, originally for use in the game of Icehouse.
  Fact 1.0: The Looney Pyramids game system was invented by Andrew Looney.
  Fact 1.1: The Looney Pyramids game system was invented by John Cooper.
  Fact 1.2: The Looney Pyramids game system was invented in 1987.
  Fact 1.3: The Looney Pyramids game system was originally for use in the game of Icehouse.
Sentence 2:  In 1998, "Kill Doctor Lucky" won the Origins Award for "Best Abstract Board Game of 1997".
  Fact 2.0: "Kill Doctor Lucky" won the Origins Award in 1998.
  Fact 2.1: "Kill Doctor Lucky" won the Origins Award for "Best Abstract Board Game of 1997".

Retrieved items:
<item 1>
Hidden Expedition
Hidden Expedition is a series of single player hidden object casual games developed by the internal studios of Big Fish Games for the first five installments
(using Big Fish Games Framework as the engine, and with the help of Flood Light Games in the 5th game), and by Eipix Entertainment for all subsequent
installments.

 As of September 2017, a total of fifteen games in the series have been released.

 The Hidden Expedition series marks the second major hidden object game brand from Big Fish Games.

 The second game in the series, "Hidden Expedition: Everest", would go on to be the first game Big Fish released to the iPhone.
</item 1>

<item 2>
Icehouse pieces
Icehouse pieces, or Icehouse Pyramids, Treehouse pieces, Treehouse Pyramids and officially Looney Pyramids, are nestable and stackable pyramid-shaped gaming
pieces and a game system.

 The game system was invented by Andrew Looney and John Cooper in 1987, originally for use in the game of Icehouse.
</item 2>

<item 3>
Kill Doctor Lucky
Kill Doctor Lucky is a humorous board game designed by James Ernest and released in 1996 by Cheapass Games.

 In 1998, "Kill Doctor Lucky" won the Origins Award for "Best Abstract Board Game of 1997".
</item 3>

<item 4>
Icehouse (song)
"Icehouse" is a song by the Australian rock band Flowers.

 It was released as a single in Europe in 1982 by Chrysalis Records from the band's first album, "Icehouse", after the band changed its name to Icehouse.

 In the United States, the song peaked at number 28 on the "Billboard" Top Tracks chart in 1981.
</item 4>

<item 5>
Treehouse (game)
Treehouse is a game in which players try to get their configuration of Icehouse pieces to match the central configuration, shared by all players.

 The rolling of the special "Treehouse Die" tells the player what kind of move to make to change his own or the central configuration, and then he does so to
best move towards the goal.
</item 5>

<item 6>
No Promises (Icehouse song)
"No Promises" is the first single released by Australian band, Icehouse from the band's 1986 album, "Measure for Measure" and was released in November 1985 on
Regular Records as 7" Vinyl Single and 12" Vinyl Single formats.

 It was also released in the UK and Europe by Chrysalis Records again on 7" and 12" formats but with different track listings.

 It was subsequently released in the US by Chrysalis on 7" and 12" formats, again with different track listings.

 It peaked at #30 on the Australian singles charts.

 A remix version by [love] tatto was released on the Icehouse album "Meltdown" in 2002.
</item 6>

<item 7>
Boxes (Icehouse album)
Boxes is a soundtrack album by Australian band Icehouse, released by Festival Records / Chrysalis Records in November 1985.

 The work was originally conceived by its composers, Iva Davies and Robert Kretschmer, in collaboration with choreographer Graeme Murphy of the Sydney Dance
Company for performance as the ballet "Boxes".

 The first live performance of "Boxes" was given by the Sydney Dance Company together with Davies and Kretschmer of Icehouse and guest percussionist Masaki
Tanazawa in the Opera Theatre of the Sydney Opera House on 7 November 1985.

 "Boxes" was released in the US under the name Sydney Dance Company.

 The single "No Promises" was released as a 12" in November and a cassingle in December on Regular Records for Australian and New Zealand markets, it contained
other tracks from "Boxes"; two tracks from "Boxes" were later released on their next studio album "Measure for Measure" in April 1986.
</item 7>

<item 8>
Can't Help Myself
"Can't Help Myself" is the first single released by the Australian synthpop/rock band Flowers, later known as Icehouse.

 It was released in May 1980 as a 7" vinyl single on independent label, Regular Records, five months ahead of debut album "Icehouse".

 A 10" vinyl single was released in July and had a cover depicting individual images of band members diagonally across the band's name and the single's title
(see infobox at right middle).

 It peaked at #10 on the Australian Singles Charts.
</item 8>

<item 9>
Love in Motion (song)
"Love in Motion" is the first new material released by the Australian rock synthpop band Icehouse as a 7" vinyl single-only in October 1981 on Regular Records
for the Australian market.

 The band had been known as Flowers until 27 June 1981 after which they changed their name to Icehouse, they had signed to Chrysalis Records and most of
Flowers' material was released under the new name into Europe, UK and US markets.

 "Love in Motion" peaked at #10 on the Australian singles charts.

 The B-side, "Goodnight, Mr. Matthews" was included on the 1982 album "Primitive Man" with "Love in Motion" included on the Chrysalis Records US / European
versions of the album, the UK 1983 version of the album was re-titled "Love in Motion".
</item 9>

<item 10>
The Hunter (EP)
The Hunter is the first EP (and follow up to the 2010 solo album "The Boxer") by Kele Okereke (under the professional name Kele), lead singer and rhythm
guitarist of the British rock band Bloc Party.

 It was released on 7 November 2011 by Wichita Recordings in the UK, set back a week from the original release date, and 3 days earlier on 4 November 2011 by
Wichita Recordings and Liberator Music in Australia.

 The first single released from the EP was "What Did I Do?

", which features guest vocals from Lucy Taylor.

 The music video was released on 13 September 2011, with the song released as a digital single in Japan 10 days later.
</item 10>
=== END ITEM id=dc347b5f22198567 ===

=== ITEM id=546a0943c47dbec6 ===
Gold sentences and facts:
Sentence 0: Rolf Martin Theodor Dahlgren (7 July 1932 – 14 February 1987) was a Swedish-Danish botanist, professor at the University of Copenhagen from 1973 to his death.
  Fact 0.0: Rolf Martin Theodor Dahlgren was born on 7 July 1932.
  Fact 0.1: Rolf Martin Theodor Dahlgren died on 14 February 1987.
  Fact 0.2: Rolf Martin Theodor Dahlgren was a Swedish-Danish botanist.
  Fact 0.3: Rolf Martin Theodor Dahlgren was a professor at the University of Copenhagen.
  Fact 0.4: Rolf Martin Theodor Dahlgren was a professor at the University of Copenhagen from 1973 until his death.
Sentence 1:  The university has 23,473 undergraduate students, 17,398 postgraduate students, 2,968 doctoral students and over 9,000 employees.
  Fact 1.0: The university has 23,473 undergraduate students.
  Fact 1.1: The university has 17,398 postgraduate students.
  Fact 1.2: The university has 2,968 doctoral students.
  Fact 1.3: The university has over 9,000 employees.

Retrieved items:
<item 1>
Ebba Witt-Brattström
Ebba Witt-Brattström (born 1953) is a Swedish scholar in comparative literature.

 She is Professor of Literature and head of department at Södertörn University outside Stockholm, and a well-known feminist.

 In the 1970s she was a member of the feminist organisation Grupp 8, and in 2005 she was one of the founders of the feminist political organisation and party
Feministiskt Initiativ, although she later distanced herself from the organisation and criticized its alleged strong left-wing tendencies.
</item 1>

<item 2>
University of Copenhagen
The University of Copenhagen (UCPH) (Danish: "Københavns Universitet" ) is the oldest university and research institution in Denmark.

 Founded in 1479 as a studium generale, it is the second oldest institution for higher education in Scandinavia after Uppsala University (1477).

 The university has 23,473 undergraduate students, 17,398 postgraduate students, 2,968 doctoral students and over 9,000 employees.

 The university has four campuses located in and around Copenhagen, with the headquarters located in central Copenhagen.

 Most courses are taught in Danish; however, many courses are also offered in English and a few in German.

 The university has several thousands of foreign students, about half of whom come from Nordic countries.
</item 2>

<item 3>
Dimitrije Đorđević (historian)
Đorđević was awarded his doctorate in 1962.

 In 1970, Đorđević took up a position as a Full Professor of History at the University of California, Santa Barbara, joining a strong faculty in European
History including Joachim Remak, Leonard Marsak, and C. Warren Hollister.

 He was elected a member of the Serbian Academy of Science and Arts in 1985.

 A popular undergraduate lecturer and graduate mentor at the University of California, Santa Barbara, in 1992 many of his former students contributed to his
"Festschrift" entitled "Scholar, Patriot, Mentor: Historical Essays in Honor of Dimitrije Djordjevic".

 In retirement, Đorđević published his autobiography, "Scars and Memory: Four Lives in One Lifetime", describing his World War II and post World War II
experiences.

 Professor Đorđević died in Santa Barbara on March 5, 2009.
</item 3>

<item 4>
Rolf Dahlgren
Rolf Martin Theodor Dahlgren (7 July 1932 – 14 February 1987) was a Swedish-Danish botanist, professor at the University of Copenhagen from 1973 to his death.
</item 4>

<item 5>
Duncan Haldane
Frederick Duncan Michael Haldane {'1': ", '2': ", '3': ", '4': "} (born 14 September 1951), known as F. Duncan Haldane, is a British born physicist who is
Eugene Higgins Professor of Physics at the physics department of Princeton University, and a Distinguished Visiting Research Chair at Perimeter Institute for
Theoretical Physics.

 He won the 2016 Nobel Prize in Physics with David J. Thouless and John Michael Kosterlitz.
</item 5>

<item 6>
Rolf M. Zinkernagel
Rolf Martin Zinkernagel {'1': ", '2': ", '3': ", '4': "} (born January 6, 1944 in Riehen, Basel-Stadt, Switzerland) is Professor of Experimental Immunology at
the University of Zurich.

 He was awarded the Nobel Prize in Physiology or Medicine in 1996 for the discovery of how the immune system recognizes virus-infected cells.
</item 6>

<item 7>
Dalhousie University
Dalhousie University (commonly known as Dal) is a public research university in Nova Scotia, Canada, with three campuses in Halifax, a fourth in Bible Hill, and
medical teaching facilities in Saint John, New Brunswick.

 Dalhousie offers more than 4,000 courses and 180 degree programs in twelve undergraduate, graduate, and professional faculties.

 The university is a member of the U15, a group of research-intensive universities in Canada.
</item 7>

<item 8>
Theodor von Heuglin
Martin Theodor von Heuglin (20 March 1824, Hirschlanden, Württemberg – 5 November 1876), was a German explorer and ornithologist.
</item 8>

<item 9>
Gertrud Dahlgren
Gertrud Dahlgren (1931–2009) was a Swedish botanist.

 She was born at Klippan in Scania, and after graduating from university at Helsingborg went to the University of Lund for graduate studies.

 There she obtained her M.Sc.

 in chemistry and biology.

 She was married to fellow botanist Rolf Dahlgren (1932–1987), who was killed in a car accident.

 The couple had three children, Susanna, Helena and Fredrik.

 She died in December 2009 at the age of 78.
</item 9>

<item 10>
Ekhard Salje
Ekhard Karl Hermann Salje, FRS (born 1946) is Professor of Mineralogy and Petrology and former Head of the Department of Earth Sciences, Cambridge University.
</item 10>
=== END ITEM id=546a0943c47dbec6 ===

=== ITEM id=bd411d84170d4b00 ===
Gold sentences and facts:
Sentence 0: David Ross Lederman (December 12, 1894 – August 24, 1972) was an American film director noted for his Western/action/adventure films of the 1930s and 1940s.
  Fact 0.0: David Ross Lederman was born December 12, 1894.
  Fact 0.1: David Ross Lederman died August 24, 1972.
  Fact 0.2: David Ross Lederman was an American film director.
  Fact 0.3: David Ross Lederman was noted for his Western, action, and adventure films of the 1930s and 1940s.
Sentence 1: Stefan Ruzowitzky (born 25 December 1961) is an Austrian film director and screenwriter.
  Fact 1.0: Stefan Ruzowitzky was born 25 December 1961.
  Fact 1.1: Stefan Ruzowitzky is an Austrian film director.
  Fact 1.2: Stefan Ruzowitzky is an Austrian screenwriter.

Retrieved items:
<item 1>
Howard Rosenman
Howard Rosenman (born February 1, 1945), also known as Zvi Howard Rosenman, is an American producer and motion picture executive.

 He specializes in producing romantic comedy films and documentary films.

 Some of his most popular productions include "Father of the Bride" (1991) starring Steve Martin and Diane Keaton, Joss Whedon's "Buffy the Vampire Slayer"
(1992) and "The Family Man" (2000) starring Nicolas Cage.

 Rosenman's documentary film "" won the Peabody Award and the 1990 Academy Award for Best Documentary Feature; his film "The Celluloid Closet" also won the
Peabody Award.
</item 1>

<item 2>
Elem Klimov
Elem Germanovich Klimov (Russian: Эле́м Ге́рманович Кли́мов ; 9 July 1933 – 26 October 2003) was a Soviet Russian film director.

 He studied at VGIK, and was married to film director Larisa Shepitko.

 Klimov is best known in the West for his final film, 1985's "Come and See" ("Иди и смотри"), which follows a teenage boy in German-occupied Belarus during the
German-Soviet War and is often considered one of the greatest war films ever made.

 He also directed dark comedies, children's movies, and historical pictures.
</item 2>

<item 3>
Devid Striesow
Devid Striesow (born 1 October 1973 in Bergen auf Rügen) is a German actor.

 He starred as "Sturmbannführer Herzog" (Bernhard Krüger) in Stefan Ruzowitzky's 2007 film "The Counterfeiters", which was awarded the Academy Award for Best
Foreign Language Film for that year.
</item 3>

<item 4>
Brian Ralston
Brian Ralston (born April 12, 1974) is a composer and musician living in Los Angeles.

 Ralston is a graduate of the University of Arizona and the USC Thornton School of Music Scoring for Motion Pictures and Television program.

 Brian's latest film is the 2017 drama Rose starring Cybill Shepherd, James Brolin and Pam Grier.

 In 2012 he scored the 20th Century Fox inspirational sports film Crooked Arrows, starring Brandon Routh, directed by Steve Rash.

 He has also composed music for the television series "Angel" (Season 4) and scores to the theatrical motion pictures "9/Tenths", directed by Bob Degus
("Pleasantville") starring Gabrielle Anwar, Henry Ian Cusick and Dave Ortiz, the Magnolia Pictures teen heist movie Graduation directed by Mike Mayer and
starring Adam Arkin, Shannon Lucio, Chris Marquette, Riley Smith and Chris Lowell, and the dramatic feature Don't Fade Away directed by Luke Kasdan, starring
Beau Bridges, Mischa Barton and Ryan Kwanten.
</item 4>

<item 5>
Lev Kuleshov
Lev Vladimirovich Kuleshov (Russian: Лев Влади́мирович Кулешо́в ; 13 January [O.S. 1 January] 1899 – 29 March 1970) was a Russian and Soviet filmmaker and film
theorist, one of the founders of the world's first film school, the Moscow Film School.

 He was given the title People's Artist of the RSFSR in 1969.

 He was intimately involved in development of the style of film making known as Soviet montage, especially it's psychological underpinning, including the use of
editing and the cut to emotionally influence the audience, a principle known as the Kuleshov effect.

 He also developed the theory of creative geography, which is the use of the action around a cut to connect otherwise disparate settings into a cohesive
narrative.
</item 5>

<item 6>
D. Ross Lederman
David Ross Lederman (December 12, 1894 – August 24, 1972) was an American film director noted for his Western/action/adventure films of the 1930s and 1940s.
</item 6>

<item 7>
Ruggero Deodato
Ruggero Deodato (born 7 May 1939) is an Italian film director, screenwriter, and actor.
</item 7>

<item 8>
Wolfgang Becker
Wolfgang Becker (born 22 June 1954) is a German film director and writer.

 He is best known to the international audience for his work "Good Bye Lenin!

" (2003).
</item 8>

<item 9>
Stefan Ruzowitzky
Stefan Ruzowitzky (born 25 December 1961) is an Austrian film director and screenwriter.
</item 9>

<item 10>
Max Ophüls
Maximillian Oppenheimer (6 May 1902 – 26 March 1957), known as Max Ophüls (] ), was a German-born film director who worked in Germany (1931–1933), France
(1933–1940 and 1950–1957), and the United States (1947–1950).

 He made nearly 30 films, the latter ones being especially notable: "La Ronde" (1950), "Le Plaisir" (1952), "The Earrings of Madame de..." (1953) and "Lola
Montès" (1955).
</item 10>
=== END ITEM id=bd411d84170d4b00 ===

=== ITEM id=4f0c860024d38f35 ===
Gold sentences and facts:
Sentence 0: The Great War is a 26-episode documentary series from 1964 on the First World War.
  Fact 0.0: The Great War is a documentary series.
  Fact 0.1: The Great War has 26 episodes.
  Fact 0.2: The Great War is a documentary series from 1964.
  Fact 0.3: The Great War is about the First World War.
Sentence 1:  The documentary was a co-production of the Imperial War Museum, the British Broadcasting Corporation, the Canadian Broadcasting Corporation and the Australian Broadcasting Commission.
  Fact 1.0: The Great War was a co-production of the Imperial War Museum.
  Fact 1.1: The Great War was a co-production of the British Broadcasting Corporation.
  Fact 1.2: The Great War was a co-production of the Canadian Broadcasting Corporation.
  Fact 1.3: The Great War was a co-production of the Australian Broadcasting Commission.
Sentence 2:  The narrator was Michael Redgrave, with readings by Marius Goring, Ralph Richardson, Cyril Luckham, Sebastian Shaw and Emlyn Williams.
  Fact 2.0: Michael Redgrave was the narrator of The Great War.
  Fact 2.1: Marius Goring gave readings in The Great War.
  Fact 2.2: Ralph Richardson gave readings in The Great War.
  Fact 2.3: Cyril Luckham gave readings in The Great War.
  Fact 2.4: Sebastian Shaw gave readings in The Great War.
  Fact 2.5: Emlyn Williams gave readings in The Great War.
Sentence 3: Imperial War Museums (IWM) is a British national museum organisation with branches at five locations in England, three of which are in London.
  Fact 3.0: Imperial War Museums (IWM) is a British national museum organisation.
  Fact 3.1: Imperial War Museums has branches at five locations in England.
  Fact 3.2: Three of the five Imperial War Museums branches are in London.

Retrieved items:
<item 1>
Imperial War Museum (disambiguation)
The Imperial War Museum is a British national museum organisation with branches at five locations in England.
</item 1>

<item 2>
A Word in Your Ear
A Word in Your Ear is a game show that originally aired BBC1 from 19 April 1993 to 14 October 1994 then on The Family Channel from 1995.

 It was hosted by Gordon Burns.

 The host presided over male and female pairs of celebrities as they participate in a few rounds of communication games.

 Celebrities appearing included Lynsey De Paul, Nick Owen, Philippa Kennedy and Bob Holness.
</item 2>

<item 3>
British National Bibliography
British National Bibliography (BNB) was established to publish a list of the books, journals and serials that are published in the United Kingdom and Republic
of Ireland.

 It also includes information on forthcoming titles.

 New books and serials have been recorded in the British National Bibliography (BNB) since 1950.

 The British National Bibliography was first established at the British Museum.

 This is the single most comprehensive listing of UK titles.

 UK and Irish publishers are obliged by legal deposit to send a copy of all new publications, including serial titles to BNB for listing.

 The BNB publish the list weekly in electronic form: the last printed weekly list appeared in December 2011.
</item 3>

<item 4>
Vital Signs (TV series)
Vital Signs is a British television drama series airing on ITV from 2006.

 It stars Tamzin Outhwaite as a supermarket check-out operator who decides to become a doctor.

 The series co-stars William Beck, Fraser Ayres, Eve Best, Claudie Blakley, Lucinda Dryzek, Beth Goddard, Alfie Hunter, Brooke Kinsella, Harry Lloyd, Peter Rnic
and Steven Waddington.

 The filming of the show is based in numerous London hospitals and medical schools; predominantly the show has been shot in St George's Hospital and Medical
School.
</item 4>

<item 5>
Black Country Living Museum
The Black Country Living Museum (formerly The Black Country Museum) is an open-air museum of rebuilt historic buildings in Dudley in the West Midlands of
England.

 It is located in the centre of the Black Country, 10 miles west of Birmingham.

 The museum occupies 105000 m2 of former industrial land partly reclaimed from a former railway goods yard, disused lime kilns, canal arm and former coal pits.
</item 5>

<item 6>
Imperial War Museum
Imperial War Museums (IWM) is a British national museum organisation with branches at five locations in England, three of which are in London.

 Founded as the Imperial War Museum in 1917, the museum was intended to record the civil and military war effort and sacrifice of Britain and its Empire during
the First World War.

 The museum's remit has since expanded to include all conflicts in which British or Commonwealth forces have been involved since 1914.

 As of 2012, the museum aims "to provide for, and to encourage, the study and understanding of the history of modern war and 'wartime experience'".
</item 6>

<item 7>
First Life (TV series)
First Life is a 2010 British nature documentary series written and presented by David Attenborough, also known by the expanded titles David Attenborough's First
Life (UK) and First Life with David Attenborough (USA).

 It was first broadcast in the USA as a two-hour special on the Discovery Channel on 24 October 2010.

 In the United Kingdom it was broadcast as a two-part series on BBC Two on 5 November 2010.

 "First Life" sees Attenborough tackle the subject of the origin of life on Earth.

 He investigates the evidence from the earliest fossils, which suggest that complex animals first appeared in the oceans around 540 million years ago, an event
known as the Cambrian Explosion.

 Trace fossils of multicellular organisms from an even earlier period, the Ediacaran biota, are also examined.

 Attenborough travels to Canada, Morocco and Australia, using some of the latest fossil discoveries and their nearest equivalents amongst living species to
reveal what life may have been like at that time.

 Visual effects and computer animation are used to reconstruct and animate the extinct life forms.
</item 7>

<item 8>
Michael Palin: Around the World in 80 Days
Around the World in 80 Days is a 7-part BBC television travel series first broadcast on BBC1 in 1989.

 It was presented by comedian and actor Michael Palin.

 The show was inspired by Jules Verne's classic novel "Around the World in Eighty Days", in which a character named Phileas Fogg accepts a wager to
circumnavigate the globe in eighty days or less.

 Palin was given the same deadline, and not allowed to use aircraft, which did not exist in Jules Verne's time and would make completing the journey far too
easy.

 He followed Phileas Fogg's route as closely as possible.

 Along the way he commented on the sights and cultures he encountered.

 Palin encountered several setbacks during his voyage, partly because he travelled with a five-person film crew, who are collectively named after Passepartout,
Phileas Fogg's manservant.
</item 8>

<item 9>
The Great War (documentary)
The Great War is a 26-episode documentary series from 1964 on the First World War.

 The documentary was a co-production of the Imperial War Museum, the British Broadcasting Corporation, the Canadian Broadcasting Corporation and the Australian
Broadcasting Commission.

 The narrator was Michael Redgrave, with readings by Marius Goring, Ralph Richardson, Cyril Luckham, Sebastian Shaw and Emlyn Williams.

 Each episode is  40 minutes long.
</item 9>

<item 10>
From the Sea to the Land Beyond
From The Sea to the Land Beyond: Britain's Coast on Film is a documentary feature film directed by Penny Woolcock, with an original soundtrack by British
indie-rock band British Sea Power.

 The project was originally produced by Crossover and Sheffield Doc/Fest as part of The Space project from the BBC and the Arts Council England.

 The film was edited by Alex Fry.
</item 10>
=== END ITEM id=4f0c860024d38f35 ===

=== ITEM id=4575970eafd8aaed ===
Gold sentences and facts:
Sentence 0:  For almost 20 years, it held the record for selling 2 million copies in the UK in the fastest time, doing so in 42 days; in 2015, Adele surpassed the record when her album "25" took just 29 days to reach the landmark.
  Fact 0.0: For almost 20 years, a record (held by a subject not named in this sentence) stood for selling 2 million copies in the UK in the fastest time.
  Fact 0.1: The record for selling 2 million copies in the UK in the fastest time was set in 42 days.
  Fact 0.2: In 2015, Adele surpassed the record.
  Fact 0.3: Adele's album '25' took just 29 days to reach the 2 million copies landmark.
Sentence 1:  At the 51st Grammy Awards in 2009, Adele received the awards for Best New Artist and Best Female Pop Vocal Performance.
  Fact 1.0: The 51st Grammy Awards were held in 2009.
  Fact 1.1: Adele received the Best New Artist award at the 51st Grammy Awards.
  Fact 1.2: Adele received the Best Female Pop Vocal Performance award at the 51st Grammy Awards.

Retrieved items:
<item 1>
Billboard Year-End
"Billboard" Year-End charts are a cumulative measure of a single or album's performance in the United States, based upon the "Billboard" magazine charts during
any given chart year.

 "Billboard's" "chart year" runs from the first Billboard "week" of December to the final week in November, but because the Billboard week is dated in advance
of publication, the last calendar week for which sales are counted is usually the third week in November.

 This altered calendar allows for "Billboard" to calculate year-end charts and release them in time for its final print issue in the last week of December.

 Prior to incorporating chart data from Nielsen SoundScan (from 1991), year-end charts were calculated by an inverse-point system based solely on a title's
performance (for example a single appearing on the "Billboard" Hot 100 would be given one point for a week spent at position 100, two points for a week spent at
position ninety-nine, and so forth, up to 100 points for each week spent at number one).

 Other factors including the total weeks a song spent on the chart and at its peak position were calculated into its year-end total.
</item 1>

<item 2>
Japan Record Award for Best New Artist
The Japan Record Award for Best New Artist (最優秀新人賞 ) is awarded annually.

 Until the 10th Japan Record Awards it was called New Artist Award, since 11th — Best New Artist Award.

 At present all nominees for the Best New Artist Award are awarded the New Artist Award.
</item 2>

<item 3>
Catfish and the Bottlemen
Catfish and the Bottlemen are a British rock band, formed in Llandudno, Wales in 2007.

 Their debut album, "The Balcony", reached number 10 in the UK Albums Chart and achieved Platinum status on 30 December 2016.

 The band have toured in South America, Japan, UK, Europe, North America, and Australia and have featured in a number of festivals including Lollapalooza,
Glastonbury, Latitude, Community festival, Reading and Leeds, T in the Park, Governors Ball and Bonnaroo.

 They won a Brit Award for British Breakthrough act on 24 February 2016.

 On 27 May 2016, they released their second album, "The Ride".
</item 3>

<item 4>
List of Billboard 200 number-one albums of 2002
The band Creed continued its eight week long run on the chart but is credited as the longest running album 2001.

 Jennifer Lopez earned her second number one album on the charts with "", which became the highest first week sales of a remix album at the time.

 R&B artist Ashanti earned her first number one album with her self-titled debut album "Ashanti", which opened up with first week sales of 503,000 copies in its
first week alone.

 Puff Daddy earned his first number one album since "No Way Out" back in 1997.

 Rapper Jay-Z earned his fifth chart topper with "", which opened up with first week sales of 545,000 copies alone.

 Heavy metal band Disturbed earned its first number one album on the chart with "Believe", which opened up with first week sales of 284,000 copies alone.

 Country music singer Shania Twain's album "Up!

"
</item 4>

<item 5>
List of Billboard 200 number-one albums of 2002
The highest-selling albums and EPs in the United States are ranked in the "Billboard" 200, published by "Billboard" magazine.

 The data are compiled by Nielsen Soundscan based on each album's weekly physical and digital sales.

 25 acts achieved number one albums during this year with artist such as Nelly and Shania Twain who had their albums debut at number one on the chart.

 Rapper Eminem's "The Eminem Show" is the best selling album of 2002 selling over approximately 7.6 million copies by the end of the year.

 It is also the longest running album of 2002 spending six non-consecutive weeks the chart and was known for its first full week of sales debut of 1.322 million
copies which Nielsen SoundScan scanned as the sixth largest sales of all time in its first week.

 Its debut of 1.322 million copies has still not been matched by any album today since except for Taylor Swift's album "1989", which opened with first week
sales of 1.279 million copies.
</item 5>

<item 6>
Adele
Adele Laurie Blue Adkins (born 5 May 1988) ( ) is an English singer-songwriter.

 After graduating from the BRIT School for Performing Arts and Technology in 2006, Adele was given a recording contract by XL Recordings after a friend posted
her demo on Myspace the same year.

 In 2007, she received the Brit Awards "Critics' Choice" award and won the BBC Sound of 2008 poll.

 Her debut album, "19", was released in 2008 to commercial and critical success.

 It is certified seven times platinum in the UK, and three times platinum in the US.

 The album contains her first song, "Hometown Glory", written when she was 16, which is based on her home suburb of West Norwood in London.

 An appearance she made on "Saturday Night Live" in late 2008 boosted her career in the US.

 At the 51st Grammy Awards in 2009, Adele received the awards for Best New Artist and Best Female Pop Vocal Performance.
</item 6>

<item 7>
Paul Epworth
Paul Richard Epworth (born 25 July 1974 in Bishop's Stortford, Hertfordshire) is an English music producer, musician, and songwriter.

 His production and writing credits include Adele, Rihanna, and Maxïmo Park amongst many others.

 On 12 February 2012 at the 54th Grammy Awards, Epworth won four Grammy Awards for Producer of the Year, Album of the Year (Adele's "21"), and Song of the Year
and Record of the Year (for "Rolling in the Deep").

 He won the Academy Award for Best Original Song alongside Adele, for "Skyfall".

 His sister Mary Epworth is a singer and songwriter.

 He is a member of the Music Producers Guild.

 He also has a record label, Wolf Tone, whose artists include Glass Animals, Rosie Lowe and Plaitum.

 He has won 'Producer of the Year' at the BRIT Awards three times, the most recent in 2015.

 At the 59th Grammy Awards in 2017, Epworth won a Grammy for Album of the Year for his work on Adele's 25".
</item 7>

<item 8>
List of Billboard 200 number-one albums of 2002
Country music singer Shania Twain's album "Up!

" opened up with a huge first week sales of 857,000 copies in its first week alone, giving her the recognition of the highest first week sales of her career and
second highest of the year, only behind Eminem's "The Eminem Show" and at the time the fastest selling solo female album ever.

 Nelly's album "Nellyville" opened up with his highest first week sales of his career which logged on with huge sales of 714,000 copies in its first week alone,
which beat his sales of his debut album "Country Grammar", which opened up with first week sales of 235,000 copies.

 Country singer Alan Jackson album "Drive" gave him his first number one album on the chart and opened up with first week sales of 211,000 copies alone.
</item 8>

<item 9>
MTV Video Music Award for Best New Artist
The MTV Video Music Award for Best New Artist have been given out since the first annual MTV Video Music Awards in 1984.

 From then to 2006, the award was named Best New Artist in a Video.

 In 2007 its name was changed to Best New Artist, as the category underwent a format change to award the artist's body of work for the full year rather than a
specific video.

 For the 2008 ceremony, though, while the award retained its 2007 name, it returned to the format of awarding a specific video rather than the artist's full
body of work.

 The category was later renamed Artist to Watch through 2013 to 2015 while still keeping the format of an award going to a certain video.
</item 9>

<item 10>
Robson &amp; Jerome (album)
Robson & Jerome is the debut studio album by British singing duo Robson & Jerome.

 It was released in the UK in 1995 by RCA Records, and reached number one on the UK Albums Chart.

 It was the Christmas number one album of 1995, and was the biggest-selling album of the year, with sales of 2,040,000.

 For almost 20 years, it held the record for selling 2 million copies in the UK in the fastest time, doing so in 42 days; in 2015, Adele surpassed the record
when her album "25" took just 29 days to reach the landmark.
</item 10>
=== END ITEM id=4575970eafd8aaed ===

=== ITEM id=95f8930c44842cf3 ===
Gold sentences and facts:
Sentence 0: John André (2 May 1750 – 2 October 1780) was a British Army officer hanged as a spy by the Continental Army during the American Revolutionary War for assisting Benedict Arnold's attempted surrender of the fort at West Point, New York to the British.
  Fact 0.0: John André lived from 2 May 1750 to 2 October 1780.
  Fact 0.1: John André was a British Army officer.
  Fact 0.2: John André was hanged as a spy by the Continental Army during the American Revolutionary War.
  Fact 0.3: John André was hanged for assisting Benedict Arnold's attempted surrender of the fort at West Point, New York, to the British.
Sentence 1: Benedict Arnold (January 14, 1741 [O.S. January 3, 1740] June 14, 1801) was a general during the American Revolutionary War, who fought for the American Continental Army, and later defected to the British Army.
  Fact 1.0: Benedict Arnold lived from January 14, 1741 (Old Style: January 3, 1740) to June 14, 1801.
  Fact 1.1: Benedict Arnold was a general during the American Revolutionary War.
  Fact 1.2: Benedict Arnold fought for the American Continental Army.
  Fact 1.3: Benedict Arnold later defected to the British Army.

Retrieved items:
<item 1>
John André
John André (2 May 1750 – 2 October 1780) was a British Army officer hanged as a spy by the Continental Army during the American Revolutionary War for assisting
Benedict Arnold's attempted surrender of the fort at West Point, New York to the British.
</item 1>

<item 2>
Battle of Blue Licks
The Battle of Blue Licks, fought on August 19, 1782, was one of the last battles of the American Revolutionary War.

 The battle occurred ten months after Lord Cornwallis's famous surrender at Yorktown, which had effectively ended the war in the east.

 On a hill next to the Licking River in what is now Robertson County, Kentucky (but was then in Kentucky County, Virginia), a force of about 50 American and
Canadian Loyalists along with 300 American Indians ambushed and routed 182 Kentucky militiamen.

 It was the last victory for the Loyalists and Natives during the frontier war.
</item 2>

<item 3>
William Farrar Smith
William Farrar Smith (February 17, 1824 – February 28, 1903), known as ‘Baldy’ Smith, was a Union general in the American Civil War, notable for attracting the
extremes of glory and blame.

 He was praised for his gallantry in the Seven Days Battles and the Battle of Antietam, but was demoted for insubordination after the disastrous defeat at the
Battle of Fredericksburg.

 As chief engineer of the Army of the Cumberland, he achieved recognition by restoring a supply line that saved that army from starvation and surrender, known
the “Cracker Line”, that helped Union troops to success in the Chattanooga Campaign in the autumn of 1863.

 Leading the first operation against Petersburg, Smith’s hesitation, possibly illness-related, cost the Union a prime opportunity for a quick end to the war,
and he was relieved of command.
</item 3>

<item 4>
Siege of Yorktown
The Siege of Yorktown, also known as the Battle of Yorktown, the Surrender at Yorktown, German Battle or the Siege of Little York, ending on October 19, 1781,
at Yorktown, Virginia, was a decisive victory by a combined force of American Continental Army troops led by General George Washington and French Army troops
led by the Comte de Rochambeau over a British Army commanded by British peer and Lieutenant General Charles Cornwallis.

 The culmination of the Yorktown campaign, the siege proved to be the last major land battle of the American Revolutionary War in the North American theater, as
the surrender by Cornwallis, and the capture of both him and his army, prompted the British government to negotiate an end to the conflict.

 The battle boosted faltering American morale and revived French enthusiasm for the war, as well as undermining popular support for the conflict in Great
Britain.
</item 4>

<item 5>
Lewis Nicola
Lewis Nicola (1717 – August 9, 1807) was an Irish-born American military officer, merchant, and writer who held various military and civilian positions
throughout his career.

 Nicola is most notable for authoring the Newburgh letter, which urged George Washington to assume a royal title.

 Born in Dublin, Ireland, Nicola had been an officer in the British Army, serving in Europe before immigrating to the Thirteen Colonies.

 Establishing a residence in Philadelphia with his family, Nicola opened a library in 1767 and was active in colonial philosophical organizations.

 As a result of his work to establish the American Philosophical Society, he was elected as one of its curators.

 When the American Revolution broke out, Nicola offered his services to the colonial government, which eventually appointed him to various positions with local
forces.
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
Benedict Arnold
Benedict Arnold (January 14, 1741 [O.S. January 3, 1740] June 14, 1801) was a general during the American Revolutionary War, who fought for the American
Continental Army, and later defected to the British Army.

 While a general on the American side, he obtained command of the fortifications at West Point, New York (which after 1802 would become the site of the U.S.
Military Academy), overlooking the cliffs at the Hudson River (upriver from British-occupied New York City), and planned to surrender them to British forces.

 This plan was exposed in September 1780.

 He was commissioned into the British Army as a brigadier general.
</item 7>

<item 8>
Battle of Germantown
The Battle of Germantown was a major engagement in the Philadelphia campaign of the American Revolutionary War.

 It was fought on October 4, 1777, at Germantown, Pennsylvania, between the British Army led by Sir William Howe, and the American Continental Army, with the
2nd Canadian Regiment, under George Washington.
</item 8>

<item 9>
Benjamin Logan
Benjamin Logan (c.1742 – December 11, 1802) was an American pioneer, soldier, and politician from Shelby County, Kentucky.

 As colonel of the Kentucky County, Virginia militia during the American Revolutionary War, he was second-in-command of all the militia in Kentucky.

 He was also a leader in Kentucky's efforts to become a state.

 His brother, John Logan, was the first state treasurer of Kentucky.
</item 9>

<item 10>
Benjamin Lincoln
Benjamin Lincoln (January 24, 1733 (O.S. January 13, 1732) – May 9, 1810) was an American army officer.

 He served as a major general in the Continental Army during the American Revolutionary War.

 Lincoln is notable for being involved in three major surrenders during the war: his participation in the Battles of Saratoga (sustaining a wound shortly
afterward) contributed to John Burgoyne's surrender of a British army, he oversaw the largest American surrender of the war at the 1780 Siege of Charleston,
and, as George Washington's second in command, he formally accepted the British surrender at Yorktown.
</item 10>
=== END ITEM id=95f8930c44842cf3 ===

=== ITEM id=95e8b4f4d8ff026a ===
Gold sentences and facts:
Sentence 0:  After the Blackhawks jumped to an early 4–1 lead in the first game of the series, Mario Lemieux and the Penguins came back to win the game, sweep the series in four games, and win their second consecutive and second overall Stanley Cup.
  Fact 0.0: The Blackhawks jumped to an early 4–1 lead in the first game of the series.
  Fact 0.1: Mario Lemieux and the Penguins came back to win the first game of the series.
  Fact 0.2: Mario Lemieux and the Penguins swept the series in four games.
  Fact 0.3: The Penguins won their second consecutive Stanley Cup.
  Fact 0.4: The Penguins won their second overall Stanley Cup.
Sentence 1: Mario Lemieux, OC, CQ ( ; ] ; born October 5, 1965) is a Canadian former professional ice hockey player and current owner of the Pittsburgh Penguins team.
  Fact 1.0: Mario Lemieux was born on October 5, 1965.
  Fact 1.1: Mario Lemieux has the honorifics OC and CQ.
  Fact 1.2: Mario Lemieux is a Canadian.
  Fact 1.3: Mario Lemieux is a former professional ice hockey player.
  Fact 1.4: Mario Lemieux is the current owner of the Pittsburgh Penguins team.

Retrieved items:
<item 1>
1984 Stanley Cup Finals
The 1984 Stanley Cup Final was held between the Edmonton Oilers and the then-defending champion New York Islanders.

 The Islanders had swept the Oilers in four straight games to win the Cup.

 In 1984, the Islanders were seeking their fifth consecutive Stanley Cup championship, but the upstart Oilers would win the best-of-seven series four games to
one to win their first Stanley Cup, becoming the third post-1967 expansion team and first former World Hockey Association team to win the Cup, and also the
first team based west of Chicago to win the Cup since the WCHL's Victoria Cougars became the last non-NHL team to win it in .

 It was also the fifth straight Finals of teams that joined the NHL in 1967 or later and a rematch of the 1983 Finals—a Stanley Cup Finals rematch would not
happen again until the Finals.

 s of 2017 , the Islanders' four consecutive Cup wins (, , , 1983) and their appearance in the 1984 Cup Finals is an NHL record of 19 consecutive playoff series
wins that currently stands unbroken.
</item 1>

<item 2>
1992 Stanley Cup Finals
The 1992 Stanley Cup Finals NHL championship series was contested by the Prince of Wales Conference and defending Stanley Cup champion Pittsburgh Penguins and
the Clarence Campbell Conference champion Chicago Blackhawks.

 The Blackhawks were appearing in their first Finals since .

 After the Blackhawks jumped to an early 4–1 lead in the first game of the series, Mario Lemieux and the Penguins came back to win the game, sweep the series in
four games, and win their second consecutive and second overall Stanley Cup.

 It was the 99th year of the Stanley Cup, and the first to extend into the month of June.

 It was the last final for Chicago Stadium as it closed in 1994.
</item 2>

<item 3>
1992 Spengler Cup
The 1992 Spengler Cup was held in Davos, Switzerland, between December 26, 1992, and December 31, 1992.

 All matches were played at HC Davos's home arena, Eisstadion Davos.

 The final was won 6-5 by Team Canada over Färjestad BK.
</item 3>

<item 4>
1998 Stanley Cup Finals
Detroit coach Scotty Bowman won his eighth Stanley Cup in that capacity (having previously done so with the Montreal Canadiens in , , , , and , the Pittsburgh
Penguins in , and the Wings the previous year), tying him with former Canadiens coach Toe Blake for the record of most Cups won by a coach (which he would break
when he helped the Red Wings win the 2002 Cup).
</item 4>

<item 5>
1998 Stanley Cup Finals
The 1998 Stanley Cup Finals was the championship series of the National Hockey League's (NHL) 1997–1998 season, and the culmination of the 1998 Stanley Cup
playoffs.

 It was contested by the Western Conference champion and defending Stanley Cup champion Detroit Red Wings and the Eastern Conference champion Washington
Capitals.

 It was the 105th year of the Stanley Cup being contested.

 The series was the Capitals' first appearance in a Stanley Cup Final since the franchise's inception in 1974.

 The Red Wings won the series for the second year in a row, four games to none.

 It was the Wings' ninth Stanley Cup, and the most recent time when a Finals concluded with a sweep (as of 2017).

 This was also the last time until that a Stanley Cup Finals ended after an NBA Finals in the same season had concluded.
</item 5>

<item 6>
The material surveys hockey history and milestones: the NHL formed in 1917 after the NHA’s collapse and became the Stanley Cup’s sole major-league competitor.
Montreal Canadiens, founded in 1909, have won a record 24 Cups. Finals highlights include Montreal’s 1931 repeat; Islanders’ 1983 fourth straight title,
followed by Edmonton’s first Cup in 1984; Pittsburgh’s 1992 sweep; and Detroit’s 1998 sweep of Washington. Mario Lemieux starred
</item 6>

<item 7>
1917–18 Toronto Hockey Club season
The 1917–18 Toronto Hockey Club season was the first season of the new "Toronto" franchise in the newly organized National Hockey League (NHL).

 The team was intended as a 'temporary' franchise, operating without an official club nickname (the press would dub them the "Blue Shirts" or "Torontos", and in
1948 the NHL would engrave "Toronto Arenas" on the Stanley Cup as the 1917–18 winner) and without a formal organization separate from the Toronto Arena Company
that managed the Arena Gardens.

 Despite this, the team came together to win the first NHL Championship, competing against existing teams that had transferred directly from the National Hockey
Association (NHA).

 Toronto would go on to win the Stanley Cup by defeating the Pacific Coast Hockey Association champion Vancouver Millionaires – the first Stanley Cup for an NHL
team and the second Cup for a Toronto team after the Toronto Blueshirts' victory in the 1913–14 season of the NHA.
</item 7>

<item 8>
1983 Stanley Cup Finals
The 1983 Stanley Cup Finals was contested by the Edmonton Oilers in their first-ever Finals appearance and the defending champion New York Islanders, in their
fourth, and fourth consecutive, Finals appearance.

 The Islanders would win the best-of-seven series four games to none, to win their fourth-straight and fourth-overall Stanley Cup.

 It was also the fourth straight Finals of post-1967 expansion teams, and the first involving a former World Hockey Association (WHA) team.

 This is also the most recent time that a defending Stanley Cup champion has won the cup four years in a row, and also the first (and, to date, only) time a
North American professional sports team has won four consecutive titles in any league competition with more than twenty teams.

 Since 1983, no professional sports team on the continent has managed to win four straight championships and no NHL team has won more than two consecutive
championships (most recently the Pittsburgh Penguins in and ).
</item 8>

<item 9>
Mario Lemieux
Mario Lemieux, OC, CQ ( ; ] ; born October 5, 1965) is a Canadian former professional ice hockey player and current owner of the Pittsburgh Penguins team.

 He played parts of 17 seasons in the National Hockey League's (NHL) with the Pittsburgh Penguins, between 1984 and 2006.

 Dubbed "The Magnificent One" or "Le Magnifique" (as well as "Super Mario"), he is widely acknowledged to have been one of the best players of all time.

 He is the only player to score one goal in each of the five possible situations in a single NHL game, a feat he accomplished in 1988.

 A gifted playmaker and fast skater despite his large size, Lemieux often beat defencemen with fakes and dekes.
</item 9>

<item 10>
History of the National Hockey League
The history of the National Hockey League begins with the end of its predecessor league, the National Hockey Association (NHA), in 1917.

 After unsuccessfully attempting to resolve disputes with Eddie Livingstone, owner of the Toronto Blueshirts, executives of the three other NHA franchises
suspended the NHA, and formed the National Hockey League (NHL), replacing the Livingstone team with a temporary team in Toronto, the Arenas.

 The NHL's first quarter-century saw the league compete against two rival major leagues—the Pacific Coast Hockey Association and Western Canada Hockey
League—for players and the Stanley Cup.

 The NHL first expanded into the United States in 1924 with the founding of the Boston Bruins, and by 1926 consisted of ten teams in Ontario, Quebec, the Great
Lakes region, and the Northeastern United States.

 At the same time, the NHL emerged as the only major league and the sole competitor for the Stanley Cup; in 1947, the NHL completed a deal with the Stanley Cup
trustees to gain full control of the Cup.

 The NHL's footprint spread across Canada as Foster Hewitt's radio broadcasts were heard coast-to-coast starting in 1933.
</item 10>
=== END ITEM id=95e8b4f4d8ff026a ===

=== ITEM id=4c17304ee3821745 ===
Gold sentences and facts:
Sentence 0: Rick Husband Amarillo International Airport (IATA: AMA, ICAO: KAMA, FAA LID: AMA) is a public airport six miles (10 km) east of downtown Amarillo, in Potter and Randall Counties, Texas, United States.
  Fact 0.0: Rick Husband Amarillo International Airport has the IATA code AMA.
  Fact 0.1: Rick Husband Amarillo International Airport has the ICAO code KAMA.
  Fact 0.2: Rick Husband Amarillo International Airport has the FAA LID AMA.
  Fact 0.3: Rick Husband Amarillo International Airport is a public airport.
  Fact 0.4: Rick Husband Amarillo International Airport is six miles (10 km) east of downtown Amarillo.
  Fact 0.5: Rick Husband Amarillo International Airport is in Potter and Randall Counties, Texas, United States.
Sentence 1: Grand Forks International Airport (IATA: GFK, ICAO: KGFK, FAA LID: GFK) is a public airport five miles (8 km) northwest of Grand Forks, in Grand Forks County, North Dakota.
  Fact 1.0: Grand Forks International Airport has the IATA code GFK.
  Fact 1.1: Grand Forks International Airport has the ICAO code KGFK.
  Fact 1.2: Grand Forks International Airport has the FAA LID GFK.
  Fact 1.3: Grand Forks International Airport is a public airport.
  Fact 1.4: Grand Forks International Airport is five miles (8 km) northwest of Grand Forks.
  Fact 1.5: Grand Forks International Airport is in Grand Forks County, North Dakota.

Retrieved items:
<item 1>
Rick Husband Amarillo International Airport
Rick Husband Amarillo International Airport (IATA: AMA, ICAO: KAMA, FAA LID: AMA) is a public airport six miles (10 km) east of downtown Amarillo, in Potter and
Randall Counties, Texas, United States.

 The airport was renamed in 2003 after NASA astronaut and Amarillo native Rick Husband, who died in the Space Shuttle "Columbia" disaster in February of that
year.
</item 1>

<item 2>
Emerado, North Dakota
Emerado is a city in Grand Forks County, North Dakota, United States located near Grand Forks Air Force Base.

 It is part of the "Grand Forks, ND-MN Metropolitan Statistical Area" or "Greater Grand Forks".

 The population was 414 at the 2010 census.

 Grand Forks Air Force Base is located near Emerado.
</item 2>

<item 3>
Texas Air &amp; Space Museum
The Texas Air & Space Museum is an aviation museum located near Rick Husband Amarillo International Airport in Amarillo, Texas.

 The museum displays civilian and military aircraft as well as a wide range of air and space artifacts.
</item 3>

<item 4>
Grand Forks Municipal Airport
Grand Forks Municipal Airport is a former airport, located within current city limits approximately 3 mi west-northwest of central Grand Forks, North Dakota.

 It was closed shortly after World War II and is now redeveloped as part of the urban area of Grand Forks.
</item 4>

<item 5>
Grand Sky
Grand Sky is an UAS (unmanned aerial systems)-specific business and aviation park located at Grand Forks Air Force Base near Grand Forks, North Dakota.

 A groundbreaking for the development was held in September 2015, making it the first UAS business and aviation park in the U.S.

 The park consists of 217 acres and includes access to the Grand Forks Air Force Base runway for large UAS flights.
</item 5>

<item 6>
Grand Forks International Airport
Grand Forks International Airport (IATA: GFK, ICAO: KGFK, FAA LID: GFK) is a public airport five miles (8 km) northwest of Grand Forks, in Grand Forks County,
North Dakota.

 GFK has no scheduled passenger flights out of the country but has an "international" title (like many other airports) because it has customs service for
arrivals from Canada and other countries.
</item 6>

<item 7>
Salt Lake City International Airport
Salt Lake City International Airport (IATA: SLC, ICAO: KSLC, FAA LID: SLC) is a civil-military airport located about 4 miles west of Downtown Salt Lake City,
Utah in the United States.

 The airport is the closest commercial airport for more than 2.5 million people and is within a 30-minute drive of nearly 1.3 million jobs.
</item 7>

<item 8>
Brainerd Lakes Regional Airport
Brainerd Lakes Regional Airport (IATA: BRD, ICAO: KBRD, FAA LID: BRD) is a public use airport located three nautical miles (6 km) northeast of the central
business district of Brainerd, a city in Crow Wing County, Minnesota, United States.

 The airport is owned by the city and county.

 It is mostly used for general aviation but is also served by one commercial airline.
</item 8>

<item 9>
Downtown Grand Forks
Downtown Grand Forks is the original commercial center of Grand Forks, North Dakota.

 Located on the western bank of the Red River of the North, the downtown neighborhood is situated near the fork of the Red River and the Red Lake River.

 While downtown is no longer the dominant commercial area of the Greater Grand Forks community, it remains the historic center of Grand Forks.

 An 80.4 acre portion was listed on the National Register of Historic Places in 2005, as Downtown Grand Forks Historic District.

 Today, downtown Grand Forks is home to many offices, stores, restaurants, and bars.
</item 9>

<item 10>
Grand Forks International
The Grand Forks International (GFI) is an annual international invitational baseball tournament hosted at James Donaldson Park in Grand Forks, British Columbia.

 The GFI is the largest invitational baseball tournament in Canada and is a large part of the culture in Grand Forks.

 The tournament is also notable for being run entirely by volunteers.
</item 10>
=== END ITEM id=4c17304ee3821745 ===

=== ITEM id=e4a4ebf2cc18af40 ===
Gold sentences and facts:
Sentence 0: Kenneth William "Ken" Kwapis (born August 17, 1957) is an American film and television director and screenwriter.
  Fact 0.0: Kenneth William Kwapis, also known as Ken Kwapis, was born on August 17, 1957.
  Fact 0.1: Ken Kwapis is an American film director.
  Fact 0.2: Ken Kwapis is an American television director.
  Fact 0.3: Ken Kwapis is an American screenwriter.
Sentence 1:  He specialized in the single-camera sitcom in the 1990s and 2000s and has directed feature films such as "Sesame Street Presents Follow That Bird" (1985), "The Sisterhood of the Traveling Pants" (2005), and "He's Just Not That into You" (2009).
  Fact 1.0: He specialized in the single-camera sitcom in the 1990s.
  Fact 1.1: He specialized in the single-camera sitcom in the 2000s.
  Fact 1.2: He has directed feature films.
  Fact 1.3: He directed Sesame Street Presents Follow That Bird, released in 1985.
  Fact 1.4: He directed The Sisterhood of the Traveling Pants, released in 2005.
  Fact 1.5: He directed He's Just Not That into You, released in 2009.
Sentence 2:  As a filmmaker, he has directed films such as Valle de Lágrimas (2006) and Peace Grows (2004).
  Fact 2.0: As a filmmaker, he has directed Valle de Lágrimas, released in 2006.
  Fact 2.1: As a filmmaker, he has directed Peace Grows, released in 2004.

Retrieved items:
<item 1>
John Bailey (cinematographer)
John Ira Bailey, A.S.C. (born August 10, 1942) is an American cinematographer and film director best known for his collaborations with directors Paul Schrader,
Lawrence Kasdan, Michael Apted, and Ken Kwapis.

 In August 2017, Bailey was elected President of the Academy of Motion Picture Arts and Sciences for a four-year term.
</item 1>

<item 2>
Paul Schrader
Paul Joseph Schrader (born July 22, 1946) is an American screenwriter, film director, and film critic.

 Schrader wrote or co-wrote screenplays for four Martin Scorsese films: "Taxi Driver" (1976), "Raging Bull" (1980), "The Last Temptation of Christ" (1988), and
"Bringing Out the Dead" (1999).

 Schrader has also directed 18 feature films, including his directing debut crime drama, "Blue Collar" (co-written with his brother, Leonard Schrader), the
crime drama "Hardcore" (a loosely autobiographical film also written by Schrader), his 1982 remake of the horror classic "Cat People", the crime drama "American
Gigolo" (1980), the biographical drama "" (1985), the cult film "Light Sleeper" (1992), the drama "Affliction" (1997), the biographical film "Auto Focus"
(2002), and the erotic dramatic thriller "The Canyons" (2013).
</item 2>

<item 3>
Wong Jing
Wong Jing ( born 3 May 1955) is a Hong Kong film director, producer, actor, presenter, and screenwriter.

 A prolific filmmaker with strong instincts for crowd-pleasing and publicity, Wong Jin has played a prominent role in the Hong Kong cinema of the last
quarter-century.
</item 3>

<item 4>
Jon Paul Puno
Jon Paul Puno (born 1985) is an American filmmaker and classical crossover singer.

 As a filmmaker, he has directed films such as Valle de Lágrimas (2006) and Peace Grows (2004).

 And also played a small role in the 2006 movie I Will Always Love You (film).
</item 4>

<item 5>
Chris Gorak
Chris Gorak is an American film director who started as an art director and production designer.

 He directed the 2006 film "Right at Your Door" and the 2011 film "The Darkest Hour".

 Gorak began working in the film industry in the 1990s as an art director.

 In the 2000s, he worked on several films as production designer before becoming a film director.
</item 5>

<item 6>
Peter Guber
Howard Peter Guber (born March 1, 1942) is an executive, entrepreneur, educator, and author.

 He is Chairman and CEO of Mandalay Entertainment.

 Guber's most recent films from Mandalay Entertainment include "The Kids Are All Right", "Soul Surfer", and "Bernie".

 He has also produced "Batman", "The Witches of Eastwick", and "Flashdance."

 Guber's films have earned over $3 billion worldwide and 50 Academy Award nominations.
</item 6>

<item 7>
Takeshi Fukunaga
Takeshi Fukunaga is a Japanese filmmaker based in New York.

 His first feature film, "Out of My Hand" (2015) premiered in the Panorama section at the 65th Berlin International Film Festival and won the Top Prize at the
2015 Los Angeles Film Festival.

 The film was later released worldwide through Ava DuVernay’s distribution company, ARRAY.

 Takeshi was nominated for the John Cassavetes Award at the 2016 Independent Spirit Awards.

 In 2017, he was selected for The Residence by Cannes Film Festival's Cinéfondation to develop his second feature film.
</item 7>

<item 8>
John Dunning (film producer)
John Dunning (April 27, 1927–September 19, 2011) was a pioneering Canadian film producer from Montreal who co-founded the Canadian film production company
Cinépix and produced early works by notable Canadian directors David Cronenberg and Ivan Reitman.

 Dunning launched Cinépix with partner André Link in Montreal in the early 1960s.

 Their biggest commercial success—and the first Canadian box office hit—came with Reitman’s "Meatballs" (1979).
</item 8>

<item 9>
Ken Kwapis
Kenneth William "Ken" Kwapis (born August 17, 1957) is an American film and television director and screenwriter.

 He specialized in the single-camera sitcom in the 1990s and 2000s and has directed feature films such as "Sesame Street Presents Follow That Bird" (1985), "The
Sisterhood of the Traveling Pants" (2005), and "He's Just Not That into You" (2009).
</item 9>

<item 10>
Leonard Schrader
Leonard Schrader (November 30, 1943 – November 2, 2006) was an American screenwriter and director, most notable for his ability to write Japanese language films
and for his many collaborations with his brother, Paul Schrader.

 He earned an Academy Award Nomination for the screenplay he wrote for the film "Kiss of the Spider Woman".
</item 10>
=== END ITEM id=e4a4ebf2cc18af40 ===
