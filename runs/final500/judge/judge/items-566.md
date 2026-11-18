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

=== ITEM id=4c187a602bba002a ===
Gold sentences and facts:
Sentence 0: Mirrors is a 2008 supernatural horror film directed by Alexandre Aja, starring Kiefer Sutherland, Paula Patton, and Amy Smart.
  Fact 0.0: Mirrors is a 2008 supernatural horror film.
  Fact 0.1: Mirrors was directed by Alexandre Aja.
  Fact 0.2: Kiefer Sutherland starred in Mirrors.
  Fact 0.3: Paula Patton starred in Mirrors.
  Fact 0.4: Amy Smart starred in Mirrors.
Sentence 1: Paula Maxine Patton (born December 5, 1975) is an American actress.
  Fact 1.0: Paula Maxine Patton was born on December 5, 1975.
  Fact 1.1: Paula Maxine Patton is an American actress.

Retrieved items:
<item 1>
Heather Burns
Heather Burns (born April 7, 1975) is an American actress, known for her role as Miss Rhode Island in the 2000 film "Miss Congeniality".

 Her other film appearances include "You've Got Mail" (1998), "Two Weeks Notice" (2002) and "Bewitched" (2005).
</item 1>

<item 2>
Hitch (film)
Hitch is a 2005 American romantic comedy film directed by Andy Tennant and starring Will Smith.

 The film, which was written by Kevin Bisch, co-stars Eva Mendes, Kevin James, and Amber Valletta.

 Smith plays the main fictional character of the film, Alex "Hitch" Hitchens, who is a professional dating consultant who makes a living teaching men how to woo
women.

 The film was released on February 11, 2005 by Columbia Pictures.
</item 2>

<item 3>
Mirrors (film)
Mirrors is a 2008 supernatural horror film directed by Alexandre Aja, starring Kiefer Sutherland, Paula Patton, and Amy Smart.

 The film was first titled "Into the Mirror", but the name was later changed to "Mirrors".

 Filming began on May 1, 2007, and it was released in American theaters on August 15, 2008.
</item 3>

<item 4>
Zooey Deschanel filmography
American actress Zooey Deschanel made her film debut in the 1999 comedy feature "Mumford".

 She went on to gained public attention by co-starred in the comedy-drama "Almost Famous" (2000), the independent drama "Manic" (2001), opposite Joseph
Gordon-Levitt, and the comedy-drama "The Good Girl" (2002).

 She landed her first major role as a 18-year-old virgin in the romantic drama "All the Real Girls" (2003), for which she was nominated for an Independent
Spirit Award for Best Female Lead.

 Deschanel's biggest commercial success, as of 2017, came with the Christmas-fantasy film "Elf" (also in 2003), which grossed over $220 million worldwide.
</item 4>

<item 5>
Illusions &amp; Mirrors
Illusions & Mirrors is a 2013 short film directed by Iranian-born American artist and filmmaker Shirin Neshat.

 The film, shot in black-and-white, stars Tarek Aylouch, Michael Markiewicz, and the Israeli-born American actress Natalie Portman.

 The film was commissioned by Dior, for which Portman is a spokesmodel, for the Miss Dior Expo at the Grand Palais, Paris.

 Shirin Neshat said that the film is a tribute to the black-and-white silent films made by such surrealist filmmakers as Man Ray, Jean Cocteau, Luis Buñuel, and
Maya Deren.
</item 5>

<item 6>
Mira Nair
Mira Nair (born 15 October 1957) is an Indian American filmmaker based in New York City.

 Her production company, Mirabai Films, specializes in films for international audiences on Indian society, whether in the economic, social or cultural spheres.

 Among her best known films are "Mississippi Masala", "The Namesake", the Golden Lion-winning "Monsoon Wedding" and "Salaam Bombay!

", which was nominated for the Academy Award for Best Foreign Language Film.
</item 6>

<item 7>
In the Mirror of Maya Deren
The documentary In the Mirror of Maya Deren ("Im Spiegel der Maya Deren", 2002) is a film about avant garde filmmaker Maya Deren (1917-1961) by Austrian film
maker Martina Kudlacek.

 It is based on the biography "The Legend of Maya Deren".
</item 7>

<item 8>
License to Wed
License to Wed is a 2007 American romantic comedy film starring Robin Williams, Mandy Moore and John Krasinski, and directed by Ken Kwapis.

 The film was released in theaters on July 3, 2007.
</item 8>

<item 9>
These Old Broads
These Old Broads is a 2001 television film written by Carrie Fisher and starring her mother Debbie Reynolds, as well as Shirley MacLaine, Joan Collins, and
Elizabeth Taylor in her final film role.

 In a 2001 BBC "Omnibus" documentary about Elizabeth Taylor, Shirley MacLaine says that Julie Andrews and Lauren Bacall were originally planned to be in the
movie.

 The role of Miriam Hodges was originally offered to June Allyson.
</item 9>

<item 10>
Ela, Hela and The Hitch
Ela, Hela, and the Hitch is a play by Václav Havel.

 The play was written for the Artistic Director of the Theatre on the Balustrade, Ivan Vyskočil, as part of a longer evening, entitled "Hitchhiking".

 Along with "Ela, Hela, and the Hitch", Havel also wrote a sketch called "Motormorphosis".

 Reportedly, Vyskočil altered Havel’s sketches for the performance, though the original text was discovered by a Czech theater scholar, Lenka Jungmannová.

 "Motormorphosis", in a translation by Carol Rocamora, was performed at the Havel Festival in 2006, a world premiere of the text as written.

 "Ela, Hela, and the Hitch" premiered in an English translation by Edward Einhorn following a revival of "Motormorphosis" at New York’s Bohemian National Hall
in 2011.
</item 10>
=== END ITEM id=4c187a602bba002a ===

=== ITEM id=86324dec263d88c4 ===
Gold sentences and facts:
Sentence 0: Edmund Robert Harris (c. 1804 – 27 May 1877) was an English lawyer from Preston, Lancashire, UK who was the principal benefactor of the Harris Museum, Harris Institute or Art School, Harris Technical School and the Harris Orphanage.
  Fact 0.0: Edmund Robert Harris lived c. 1804 to 27 May 1877.
  Fact 0.1: Edmund Robert Harris was an English lawyer.
  Fact 0.2: Edmund Robert Harris was from Preston, Lancashire, UK.
  Fact 0.3: Edmund Robert Harris was the principal benefactor of the Harris Museum.
  Fact 0.4: Edmund Robert Harris was the principal benefactor of the Harris Institute or Art School.
  Fact 0.5: Edmund Robert Harris was the principal benefactor of the Harris Technical School.
  Fact 0.6: Edmund Robert Harris was the principal benefactor of the Harris Orphanage.
Sentence 1: The Harris Museum, Art Gallery & Preston Free Public Library is a Grade I listed museum building in Preston.
  Fact 1.0: The Harris Museum, Art Gallery & Preston Free Public Library is a museum building in Preston.
  Fact 1.1: The Harris Museum, Art Gallery & Preston Free Public Library is a Grade I listed building.

Retrieved items:
<item 1>
Hanbys Corner, Delaware
The last Hanby to occupy the property, Albert T. Hanby (1881-1947), another son of Samuel, attended West Chester State College before getting his law degree
from Penn Law School.

 Albert became a Philadelphia lawyer and left his farm at Hanby's Corner to be used for the good of "all the children in Delaware".

 He and his wife created a foundation in 1945 to protect the property from further development.

 Today the YMCA operates their Hanby Camp there, and the trust provide scholarships for kids who might not otherwise be able to attend.

 In 2007 that support was over $70,000.
</item 1>

<item 2>
Harris Museum
The Harris Museum, Art Gallery & Preston Free Public Library is a Grade I listed museum building in Preston.
</item 2>

<item 3>
Edmund de Unger
Edmund Robert Anthony de Unger (Hungarian: "Odon Antal Robert de Unger" , b 6 August 1918, Budapest - d 25 January 2011, Ham, Surrey) was a Hungarian-born
property developer and art collector.

 In London he built up the Keir Collection, one of the greatest post-war collections of Islamic art, bequeathed in 2008 to the Pergamon Museum of Islamic Art in
Berlin.

 The arrangement for the museum to curate the collection came to an end in July 2012.

 The collection is now hosted by the Dallas Museum of Art as of May 2014 for a 15-year renewable loan.
</item 3>

<item 4>
Edmund Robert Harris
Edmund Robert Harris (c. 1804 – 27 May 1877) was an English lawyer from Preston, Lancashire, UK who was the principal benefactor of the Harris Museum, Harris
Institute or Art School, Harris Technical School and the Harris Orphanage.
</item 4>

<item 5>
David Hildebrand Wilson
David Hildebrand Wilson is the co-founder, along with his wife, Diana Wilson, of the enigmatic Museum of Jurassic Technology in Los Angeles, California.

 After high school, Wilson enrolled at Kalamazoo College where he majored in urban entomology with a minor in art.

 He received an MFA in Experimental Animation from the California Institute of the Arts in 1976.

 He was awarded the MacArthur Fellowship and the Creative Capital Moving Image Award in 2001.

 He and his museum are the subject of the book, "Mr. Wilson's Cabinet of Wonder", published in 1995.
</item 5>

<item 6>
Dick Herrick
The Very Rev Richard William Herrick (3 December 1913 - 5 May 1981) was an eminent Anglican priest in the 20th century.

 He was educated at King Edward VI School Retford and Leeds University and was initially a civil servant.

 He was ordained after a period of study at the College of the Resurrection, Mirfield in 1940.

 he held curacies at Duston and Portsea, Portsmouth before being appointed Vicar of St Michael’s, Northampton in 1947, a post he held for a decade.

 He was then a Canon Residentiary of Chelmsford Cathedral until 1978 when he was appointed Provost of Chelmsford.

 He died in post.
</item 6>

<item 7>
Colin Hardie
Colin Graham Hardie (16 February 1906 – 17 October 1998) was a British classicist and academic.

 From 1933 to 1936, he was Director of the British School at Rome.

 From 1936 to 1973, he was a Fellow of Magdalen College, Oxford, and a tutor in classics.

 In addition, from 1967 to 1973, he was the Public Orator of the University of Oxford.

 He was a member of the Inklings, an informal literary discussion group which included the likes of J. R. R. Tolkien and C. S. Lewis.
</item 7>

<item 8>
J. Robert Donnelly Husky Heritage Sports Museum
The J. Robert Donnelly Husky Heritage Sports Museum is a public museum located on the University of Connecticut's main campus at Storrs, Connecticut.

 The museum documents and celebrates UConn's intercollegiate athletics.

 Opening its doors on January 19, 2002, the 2,700-square-foot museum was named in honor of benefactor and 1940 Connecticut basketball and football captain J.
Robert (Bob) Donnelly (1971-2005).

 Exhibits include national and regional championship trophies, trading cards, photographs, and various sports memorabilia, as well as a six-screen video wall
replaying moments of triumph for the UConn Huskies.

 A life-size fiberglass sculpture of Jonathan the Husky, UConn's mascot, greets visitors at the museum entrance.

 The Connecticut Basketball Rotunda, featuring NCAA championship trophies and life-size cutouts of Ray Allen and Rebecca Lobo, is among the museum's permanent
exhibits.
</item 8>

<item 9>
Rice Northwest Museum of Rocks and Minerals
The Rice Northwest Museum of Rocks and Minerals is a non-profit museum in Hillsboro, Oregon, United States.

 Located just north of the Sunset Highway on the northern edge of Hillsboro, the earth science museum is in the Portland metropolitan area.

 Opened in 1997, the museum’s collections date to the 1930s with the museum housed in a home built to display the rock and mineral collections of the museum
founders.

 The ranch style home is listed on the National Register of Historic Places, the first of its kind listed in Oregon.

 In 2015 the museum became a Smithsonian Affiliate museum.
</item 9>

<item 10>
Imperial War Museum (disambiguation)
The Imperial War Museum is a British national museum organisation with branches at five locations in England.
</item 10>
=== END ITEM id=86324dec263d88c4 ===

=== ITEM id=0e3a09303ec5dec0 ===
Gold sentences and facts:
Sentence 0: John Alexander (born 5 October 1955) is an English former footballer who holds the post of club secretary at Manchester United.
  Fact 0.0: John Alexander was born on 5 October 1955.
  Fact 0.1: John Alexander is English.
  Fact 0.2: John Alexander is a former footballer.
  Fact 0.3: John Alexander holds the post of club secretary at Manchester United.
Sentence 1:  Born in Liverpool, Alexander began his football career with a club called Ulysses, before being picked up by Millwall, for whom he made his league debut in 1976.
  Fact 1.0: John Alexander was born in Liverpool.
  Fact 1.1: John Alexander began his football career with a club called Ulysses.
  Fact 1.2: John Alexander was later picked up by Millwall.
  Fact 1.3: John Alexander made his league debut for Millwall in 1976.
Sentence 2:  He is the uncle of Liverpool player Trent Alexander-Arnold.
  Fact 2.0: John Alexander is the uncle of Liverpool player Trent Alexander-Arnold.
Sentence 3: Trent John Alexander-Arnold (born 7 October 1998) is an English professional footballer who plays as a right-back for Premier League club Liverpool.
  Fact 3.0: Trent John Alexander-Arnold was born on 7 October 1998.
  Fact 3.1: Trent John Alexander-Arnold is English.
  Fact 3.2: Trent John Alexander-Arnold is a professional footballer.
  Fact 3.3: Trent John Alexander-Arnold plays as a right-back.
  Fact 3.4: Trent John Alexander-Arnold plays for Premier League club Liverpool.

Retrieved items:
<item 1>
Trent Alexander-Arnold
Trent John Alexander-Arnold (born 7 October 1998) is an English professional footballer who plays as a right-back for Premier League club Liverpool.

 He is an academy graduate of Liverpool and made his senior debut for the club in 2016, aged 18.

 The following year, he won the club's Young Player of the Season award.

 He has also represented England at all youth levels from under-16 through to under-19.

 Due to the length of his name, his surname is shortened to Alexander for Premier League fixtures.
</item 1>

<item 2>
Clare Taylor
Clare Elizabeth Taylor MBE (born 22 May 1965) is an English sportswoman, the first woman to have played on a World Cup team in both cricket and football.

 She represented England at both cricket, as a member of the winning World Cup cricket team in 1993, and football (World Cup 1995).

 She was awarded the MBE in the Queen's Birthday Honours in 2000 for her services to women’s sport.

 Taylor attended Moor End High School and still has her name on the athletics records board.
</item 2>

<item 3>
Wilton Lines
Wilton Lines (fl.

 1889–1890) was an English professional footballer born in Birmingham.

 Lines played seven games for Small Heath in the inaugural 1889–90 season of the Football Alliance, covering a variety of forward positions, but was unable to
displace the established forwards.
</item 3>

<item 4>
Tommy Bell (footballer, born 1906)
Thomas Bell (9 November 1906 – 1983) was an English professional footballer born in Seaham Harbour.

 He could play at inside right, centre forward and right half and was a regular goalscorer throughout his professional career.
</item 4>

<item 5>
Billy Morgan (footballer, born 1891)
William Albert L. Morgan (3 November 1891 – after 1927) was an English professional footballer born in Old Hill, Cradley Heath, Staffordshire, who played either
at outside left or inside left.

 He played for Birmingham, Coventry City and Crystal Palace in the Football League, and was capped once for a Football League representative side.
</item 5>

<item 6>
John Logan (footballer, born 1912)
John William Logan (16 August 1912 – October 1980) was an English professional footballer born in Horden, near Peterlee, County Durham who played in the
Football League for Darlington, Barnsley and Sheffield Wednesday.

 He played as a wing half.
</item 6>

<item 7>
John Alexander (footballer, born 1955)
John Alexander (born 5 October 1955) is an English former footballer who holds the post of club secretary at Manchester United.

 Born in Liverpool, Alexander began his football career with a club called Ulysses, before being picked up by Millwall, for whom he made his league debut in
1976.

 After scoring twice in 15 appearances over the course of two years with Millwall, he moved to Reading; he enjoyed greater success with Reading, scoring nine
goals in 25 games.

 In 1981, he joined Northampton Town, but spent just one season with them before retiring from football at the age of 26 in 1982.

 Upon retiring from football, Alexander got a job with the BBC, but he later returned to football as club secretary at Watford.

 In 2000, he took up the same post at Tottenham Hotspur, spending 10 years there before applying for the same job at Manchester United, where he would replace
the retiring Ken Ramsden.

 He took over at Manchester United on 1 July 2010.

 He is the uncle of Liverpool player Trent Alexander-Arnold.
</item 7>

<item 8>
Nick Powell
Nicholas Edward "Nick" Powell (born 23 March 1994) is an English professional footballer who plays for Wigan Athletic as a midfielder.

 He began his career at Crewe Alexandra, initially as a forward, making his debut at the age of 16.

 After impressing for Crewe Alexandra during the 2011–12 season, including scoring in their 2–0 victory in the League Two play-off Final, he moved to Manchester
United in July 2012.

 Unable to break into the Manchester United first team, however, he went on loan to Wigan Athletic, Leicester City and Hull City before being released in June
2016.

 He joined Wigan Athletic in July 2016.

 He has represented England at under-16, under-17, under-18, under-19 and under-21 levels.
</item 8>

<item 9>
William McCourty
William McCourty (1884 – 10 December 1917) was an English professional footballer born in Morpeth, Northumberland, who played in the Football League for
Birmingham.

 McCourty, a former coal miner who played at left half, joined Birmingham in May 1909.

 He made his debut in the Second Division on 13 September 1909, deputising for the injured Tommy Daykin in a home game against Glossop which finished as a 2–2
draw, and was never picked again, returning to non-league football the following year.
</item 9>

<item 10>
Jack Charlton
John Charlton, OBE, DL (born 8 May 1935) is an English former footballer and manager who played as a defender.

 He was part of the England team that won the 1966 World Cup.

 He is the elder brother of former Manchester United forward Bobby Charlton, who was also a teammate in England's World Cup final victory.

 He spent his entire club career with Leeds United from 1950 to 1973, helping the club to the Second Division title (1963–64), First Division title (1968–69),
FA Cup (1972), League Cup (1968), Charity Shield (1969), Inter-Cities Fairs Cup (1968 and 1971), as well as one other promotion from the Second Division
(1955–56) and five second-place finishes in the First Division, two FA Cup final defeats and one Inter-Cities Fairs Cup final defeat.

 His 629 league and 762 total competitive appearances are club records.

 In 2006, Leeds United supporters voted Charlton into the club's greatest ever XI.
</item 10>
=== END ITEM id=0e3a09303ec5dec0 ===

=== ITEM id=edcc5e45a74be545 ===
Gold sentences and facts:
Sentence 0: Pagliacci (] ; literal translation, "Clowns") is an Italian opera in a prologue and two acts, with music and libretto by Ruggero Leoncavallo.
  Fact 0.0: Pagliacci is an Italian opera.
  Fact 0.1: Pagliacci has a prologue and two acts.
  Fact 0.2: Pagliacci has music and libretto by Ruggero Leoncavallo.
  Fact 0.3: The literal translation of Pagliacci is 'Clowns'.
Sentence 1: Suor Angelica (Sister Angelica) is an opera in one act by Giacomo Puccini to an original Italian libretto by Giovacchino Forzano.
  Fact 1.0: Suor Angelica (Sister Angelica) is an opera.
  Fact 1.1: Suor Angelica is in one act.
  Fact 1.2: Suor Angelica is by Giacomo Puccini.
  Fact 1.3: Suor Angelica has an original Italian libretto by Giovacchino Forzano.

Retrieved items:
<item 1>
Barbara Frittoli
Barbara Frittoli (born 19 April 1967) is an Italian operatic soprano who has sung leading roles in opera houses throughout Europe and in the United States.

 She was born in Milan and graduated from the Milan Conservatory.

 She made her Metropolitan Opera debut in 1995 as Micaela in "Carmen" and has gone on to sing in over 80 performances there including Donna Elvira in "Don
Giovanni", Fiordiligi in "Così fan tutte", Angelica in "Suor Angelica", Desdemona in "Otello", the title role in "Luisa Miller", Amelia in "Simon Boccanegra",
Vitellia in "La clemenza di Tito" and Alicia Ford in "Falstaff".
</item 1>

<item 2>
The Consul
The Consul is an opera in three acts with music and libretto by Gian Carlo Menotti, his first full-length opera.
</item 2>

<item 3>
Pagliacci
Pagliacci (] ; literal translation, "Clowns") is an Italian opera in a prologue and two acts, with music and libretto by Ruggero Leoncavallo.

 It is the only Leoncavallo opera that is still widely performed.

 Opera companies have frequently staged "Pagliacci" with "Cavalleria rusticana" by Mascagni, a double bill known colloquially as 'Cav and Pag'.
</item 3>

<item 4>
Stiffelio
Stiffelio is an opera in three acts by Giuseppe Verdi, from an Italian libretto by Francesco Maria Piave.

 The origin of this was the novel “Le pasteur d’hommes”, by Émile Souvestre, which was published in 1838.

 This was adapted into the French play "Le pasteur, ou L'évangile et le foyer" by Émile Souvestre and Eugène Bourgeois.

 That was in turn translated into Italian by Gaetano Vestri as "Stifellius"; this formed the basis of Piave's libretto.
</item 4>

<item 5>
Agrippina (opera)
Agrippina (HWV 6) is an "opera seria" in three acts by George Frideric Handel with a libretto by Cardinal Vincenzo Grimani.

 Composed for the 1709–10 Venice "Carnevale" season, the opera tells the story of Agrippina, the mother of Nero, as she plots the downfall of the Roman Emperor
Claudius and the installation of her son as emperor.

 Grimani's libretto, considered one of the best that Handel set, is an "anti-heroic satirical comedy", full of topical political allusions.

 Some analysts believe that it reflects Grimani's political and diplomatic rivalry with Pope Clement XI.
</item 5>

<item 6>
Bernhard Bötel
His stage repertoire included Belmonte in Wolfgang Amadeus Mozart's "Die Entführung aus dem Serail", Chapelou in Adolphe Adam's "Le postillon de Lonjumeau",
Count Almaviva in Gioachino Rossini's "The Barber of Seville", Daniel in Franz Lehár's "The Merry Widow", the Duke of Mantua in Giuseppe Verdi's "Rigoletto",
Gabriel von Eisenstein in Johann Strauss II's "Die Fledermaus", Indigo in Strauss' "Indigo und die vierzig Räuber", Jeník in Bedřich Smetana's "The Bartered
Bride", Paolino in Domenico Cimarosa's "Il matrimonio segreto", Pâris in Jacques Offenbach's "La belle Hélène", Pietro in Franz von Suppé's "Boccaccio", and
Wilhelm Meister in Ambroise Thomas' "Mignon".
</item 6>

<item 7>
Suor Angelica
Suor Angelica (Sister Angelica) is an opera in one act by Giacomo Puccini to an original Italian libretto by Giovacchino Forzano.

 It is the second opera of the trio of operas known as "Il trittico" ("Triptych").

 It received its world premiere at the Metropolitan Opera on December 14, 1918.
</item 7>

<item 8>
Lucrezia Borgia (opera)
Lucrezia Borgia is a melodramatic opera in a prologue and two acts by Gaetano Donizetti.

 Felice Romani wrote the Italian libretto after the play "Lucrezia Borgia" by Victor Hugo, in its turn after the legend of Lucrezia Borgia.

 "Lucrezia Borgia" was first performed on 26 December 1833 at La Scala, Milan.
</item 8>

<item 9>
Il trittico
Il trittico ("The Triptych") is the title of a collection of three one-act operas, "Il tabarro", "Suor Angelica", and "Gianni Schicchi", by Giacomo Puccini.

 The work received its world premiere at the Metropolitan Opera on 14 December 1918.
</item 9>

<item 10>
Rita Fornia
Rita Fornia (17 July 1878 – 27 October 1922) was an American opera singer.

 She began her career in Europe at the beginning of the 20th century singing coloratura soprano roles.

 Early on in her career her voice darkened and dropped slightly causing her to focus more within the mezzo-soprano repertoire while still singing some soprano
roles.

 She joined the Metropolitan Opera company in 1907 where she performed regularly in mostly supporting roles until her retirement in 1922.

 She is best remembered today for portraying the role of the Abbess in the original production of Puccini's "Suor Angelica" in 1918.
</item 10>
=== END ITEM id=edcc5e45a74be545 ===

=== ITEM id=c48a03ee7061ceb9 ===
Gold sentences and facts:
Sentence 0:  In return, William made Copsi earl of Northumbria and sent him back to York.
  Fact 0.0: William made Copsi earl of Northumbria in return.
  Fact 0.1: William sent Copsi back to York.
Sentence 1: York ( ) is a historic walled city at the confluence of the rivers Ouse and Foss in North Yorkshire, England.
  Fact 1.0: York is a historic walled city.
  Fact 1.1: York is at the confluence of the rivers Ouse and Foss.
  Fact 1.2: York is in North Yorkshire, England.

Retrieved items:
<item 1>
Earl Cowper
Earl Cowper (pronounced "Cooper") was a title in the Peerage of Great Britain.

 It was created in 1718 by George I for William Cowper, 1st Baron Cowper, his first Lord Chancellor, with remainder in default of male issue of his own to his
younger brother, Spencer Cowper.

 Cowper had already been created Baron Cowper of Wingham in the County of Kent, in the Peerage of England on 14 December 1706, with normal remainder to the
heirs male of his body, and was made Viscount Fordwich, in the County of Kent, at the same time as he was given the earldom, also Peerage of Great Britain and
with similar remainder.

 He was the great-grandson of William Cowper, who was created a Baronet, of Ratling Court in the County of Kent, in the Baronetage of England on 4 March 1642.

 The latter was succeeded by his grandson, the second Baronet.

 He represented Hertford in Parliament.
</item 1>

<item 2>
Two Rivers Press
Two Rivers Press is an independent publishing house, based in the English town of Reading.

 Two Rivers Press was founded in 1994 by Peter Hay (1951–2003), a local artist.

 Its name reflects his enthusiasm for the town and its two rivers, the Kennet and the Thames, and its intention to explore "the place where art and history
meet".

 The name also gives a clue to the origins of the company in the, ultimately successful, opposition to Reading's proposed "Cross-Town Route", a road scheme that
would have seriously impacted the point at which the two rivers meet.
</item 2>

<item 3>
York
York ( ) is a historic walled city at the confluence of the rivers Ouse and Foss in North Yorkshire, England.

 The municipality is the traditional county town of the historic county of Yorkshire to which it gives its name.

 The city has a rich heritage and has provided the backdrop to major political events in England throughout much of its two millennia of existence.

 The city offers a wealth of historic attractions, of which York Minster is the most prominent, and a variety of cultural and sporting activities making it a
popular tourist destination for millions.
</item 3>

<item 4>
Earl of Norbury
Earl of Norbury, in the County of Tipperary, is a title in the Peerage of Ireland.

 It was created in 1827, along with the title Viscount Glandine, of Glandine in the King's County, for the Irish politician and judge John Toler, 1st Baron
Norbury, upon his retirement as Chief Justice of the Common Pleas in Ireland.

 The titles were created with special remainder to his second son, Hector, as his eldest son, Daniel, was then considered mentally unwell.

 Lord Norbury had already been created Baron Norbury, of Ballycrenode in the County of Tipperary, in the Peerage of Ireland in 1800, with remainder to the heirs
male of his body.

 Moreover, his wife, Grace Toler (née Graham), had been created Baroness Norwood, of Knockalton in the County of Tipperary, in the Peerage of Ireland in 1797,
with remainder to the heirs male of her body.

 By the time Lord Norbury was raised to the Earldom, his wife had died and their eldest son had succeeded her as 2nd Baron Norwood.
</item 4>

<item 5>
Tripartite Indenture
The Tripartite Indenture was an agreement made in February 1405 between Owain Glyndŵr, Edmund Mortimer, and Henry Percy, 1st Earl of Northumberland, agreeing to
divide England and Wales up between them at the expense of Henry IV.

 Glyndŵr was to be given Wales, and a substantial part of the west of England, including the English portions of the Welsh Marches.

 Northumberland was to have received the north, as well as Northamptonshire, Norfolk, Warwickshire, and Leicestershire.

 The Mortimers were to have received the rest of southern England.
</item 5>

<item 6>
Two Rivers (Red River of the North)
The Two Rivers is a 7.0 mi river in Kittson County, northwestern Minnesota, in the United States.

 Formed by the North Branch of the Two Rivers and the South Branch of the Two Rivers, it is a tributary of the Red River of the North, with its outflow
traveling north through Lake Winnipeg and the Nelson River to Hudson Bay.
</item 6>

<item 7>
Earl Cowper
He represented Hertford in Parliament.

 He was succeeded by his eldest son, the aforementioned William Cowper, the third Baronet, who was elevated to the peerage as Baron Cowper in 1706 and made Earl
Cowper in 1718.

 In 1706 Lord Cowper married as his second wife Mary Clavering, daughter of John Clavering, of Chopwell, County Durham.
</item 7>

<item 8>
Copsi
Osulf, however, only ruled as earl until that autumn, when he was killed by an outlaw he was tracking.
</item 8>

<item 9>
Earl of Norbury
This son also succeeded Lord Norbury himself on his death in 1831 as 2nd Baron Norbury, whilst his younger brother Daniel succeeded to the viscountcy and
earldom according to the special remainder.

 In 1832, the second Earl also succeeded his elder brother in the two baronies.

 He had already in 1825 assumed the additional surname of Graham by Royal licence.
</item 9>

<item 10>
Copsi
Copsi (or Copsig; Old English: "Cōpsige" ) [died 1067] was a Northumbrian magnate in late Anglo-Saxon England.

 He was a supporter of Tostig, and was exiled along with him in 1065.

 Copsi soon fled to Orkney (then a part of Norway).

 The next year (1066), he joined Tostig at Sandwich, in Kent, with 17 ships.

 Copsi survived Tostig's defeat at Stamford Bridge, and when William the Conqueror prevailed at Hastings he travelled, in March 1067, to pay William homage at
Barking (where William was staying while his tower was being constructed in London).

 In return, William made Copsi earl of Northumbria and sent him back to York.

 Copsi's rule lasted a mere five weeks, at which time he was murdered by Osulf, son of Eadulf III of the ancient Bernician family which had historically
governed the area from Bamburgh, at Newburn-upon-Tyne.
</item 10>
=== END ITEM id=c48a03ee7061ceb9 ===

=== ITEM id=46c70067a510142e ===
Gold sentences and facts:
Sentence 0: The 'Burbs is a 1989 American comedy thriller film directed by Joe Dante starring Tom Hanks, Bruce Dern, Carrie Fisher, Rick Ducommun, Corey Feldman, Wendy Schaal and Henry Gibson.
  Fact 0.0: The 'Burbs is a 1989 American film.
  Fact 0.1: The 'Burbs is a comedy thriller film.
  Fact 0.2: The 'Burbs was directed by Joe Dante.
  Fact 0.3: The 'Burbs stars Tom Hanks.
  Fact 0.4: The 'Burbs stars Bruce Dern.
  Fact 0.5: The 'Burbs stars Carrie Fisher.
  Fact 0.6: The 'Burbs stars Rick Ducommun.
  Fact 0.7: The 'Burbs stars Corey Feldman.
  Fact 0.8: The 'Burbs stars Wendy Schaal.
  Fact 0.9: The 'Burbs stars Henry Gibson.
Sentence 1: Richard "Rick" Ducommun (July 3, 1952 – June 12, 2015) was a Canadian actor, comedian and writer who appeared in films and television.
  Fact 1.0: Rick Ducommun was born on July 3, 1952.
  Fact 1.1: Rick Ducommun died on June 12, 2015.
  Fact 1.2: Rick Ducommun was a Canadian actor.
  Fact 1.3: Rick Ducommun was a Canadian comedian.
  Fact 1.4: Rick Ducommun was a Canadian writer.
  Fact 1.5: Rick Ducommun appeared in films.
  Fact 1.6: Rick Ducommun appeared in television.

Retrieved items:
<item 1>
Pizza 73
Pizza 73 is a Canadian restaurant chain that offers a number of different styles of pizza, along with chicken wings.

 It has been operated by Pizza Pizza since 2007.

 Toronto-based Pizza Pizza had acquired the restaurant for a total of $CAN70.2 million.

 There are 89 locations throughout Western Canada, which include the provinces of British Columbia, Alberta, and Saskatchewan.

 The restaurant's name originates from its original phone number: 473–7373.

 Founded by David Tougas and Guy Goodwin in 1985, Pizza 73 is headquartered in Edmonton, Alberta, Canada.
</item 1>

<item 2>
Night Life (film)
Night Life is a 1989 American zombie comedy film directed by David Acomba and starring Scott Grimes, John Astin, and Cheryl Pollak.

 It centers around a high school nerd named Archie Melville who is harassed from beyond the grave by the zombie corpses of his former high school bullies
(recently deceased) after they are brought back to life by a freak lightning storm.

 It is also known as Grave Misdemeanors.
</item 2>

<item 3>
Jebb Fink
Jebb Fink is an American-Canadian stand-up comedian and television personality.

 Best known as a reporter and host for "the Big Breakfast" on A-Channel, Calgary.

 He won a Gemini as co-creator of the CBC comedy "An American in Canada" and was a weather presenter on "Your City" for City TV Calgary, he also hosted the
short-lived late night talk show "Global Late Night" for the Global Television Network in 2004.
</item 3>

<item 4>
Dunston Checks In
Dunston Checks In is a 1996 American family comedy film starring Eric Lloyd, Graham Sack, Jason Alexander, Faye Dunaway, Rupert Everett, Paul Reubens, Glenn
Shadix, and introducing Sam the Orangutan as Dunston.

 It was written by John Hopkins and Bruce Graham and directed by Ken Kwapis.
</item 4>

<item 5>
Araby Lockhart
Araby Lockhart (born December 4, 1926) is a Canadian stage and television actress, best known for her performances in the films "Capote" and "Police Academy"
and her stage performances as a member of Hart House Theatre and the Straw Hat Players.

 Lockhart has also served as President of the Actors' Fund of Canada.
</item 5>

<item 6>
The 'Burbs
The 'Burbs is a 1989 American comedy thriller film directed by Joe Dante starring Tom Hanks, Bruce Dern, Carrie Fisher, Rick Ducommun, Corey Feldman, Wendy
Schaal and Henry Gibson.

 The film was written by Dana Olsen, who also has a cameo in the movie.

 The film pokes fun at suburban environments and their eccentric dwellers.
</item 6>

<item 7>
Marty Belafsky
Marty Belafsky (born September 19, 1975) is an American actor/comedian born in Los Angeles, California.

 He began acting professionally at age 13 and was soon cast as Louis Plumb on the short-lived NBC series "Hull High".

 Shortly thereafter, Belafsky landed the role of Crutchy in the Disney musical film, "Newsies".

 He continued acting through his teens, making appearances in such television shows as "The Wonder Years", "Great Scott" and "Step By Step" and the film
"Wrestling Ernest Hemingway".

 Belafsky also voiced Kent Swanson in the video game series "Dead Rising" released in 2006.
</item 7>

<item 8>
Freebie and the Bean
Freebie and the Bean is a 1974 American action-comedy film about two off-beat police detectives who wreak havoc in San Francisco attempting to bring down a
local organized crime boss.

 The picture, a precursor to the buddy cop film genre popularized a decade later, stars James Caan, Alan Arkin, Loretta Swit and Valerie Harper.

 Harper was nominated for the Golden Globe for New Star of the Year for playing the Hispanic wife of Alan Arkin.

 The film was directed by Richard Rush.

 An article in Rolling Stone magazine alleged that Stanley Kubrick called "Freebie and the Bean" the best film of 1974.

 Arkin and Caan would not appear in another movie together until the 2008 film adaptation of "Get Smart".
</item 8>

<item 9>
Neon City
Neon City (also known as Anno 2053 in Italy, Neonski Grad in Serbia) is a 1991 Canadian post-apocalyptic science fiction film directed by Monte Markham and
written by Jeff Begun and Ann Lewis Hamilton, under the pseudonym Buck Finch.

 The cast includes Michael Ironside, Vanity, Lyle Alzado, Valerie Wildman, Nick Klar, Juliet Landau, Richard Sanders and Markham.
</item 9>

<item 10>
Vegas Vacation
Vegas Vacation is a 1997 American comedy film directed by Stephen Kessler.

 It is the fourth installment in "National Lampoon"’s "Vacation" film series, and was written by Elisa Bell, based on a story by Bell and Bob Ducsay.

 The film stars Chevy Chase, Beverly D'Angelo and Randy Quaid, with Ethan Embry and Marisol Nichols as Griswold children Rusty and Audrey.

 The film opened at #4 at the box office and grossed over $36.4 million domestically.

 "Vegas Vacation" became the first theatrical "Vacation" film not to carry the National Lampoon label or a screenwriting credit from John Hughes.
</item 10>
=== END ITEM id=46c70067a510142e ===

=== ITEM id=f90353d54d51bc56 ===
Gold sentences and facts:
Sentence 0: Dalhousie University (commonly known as Dal) is a public research university in Nova Scotia, Canada, with three campuses in Halifax, a fourth in Bible Hill, and medical teaching facilities in Saint John, New Brunswick.
  Fact 0.0: Dalhousie University, commonly known as Dal, is a public research university.
  Fact 0.1: Dalhousie University is in Nova Scotia, Canada.
  Fact 0.2: Dalhousie University has three campuses in Halifax.
  Fact 0.3: Dalhousie University has a fourth campus in Bible Hill.
  Fact 0.4: Dalhousie University has medical teaching facilities in Saint John, New Brunswick.
Sentence 1: California State Polytechnic University, Pomona (CPP, Cal Poly Pomona, or Cal Poly) is a public polytechnic university located in Pomona, California, United States.
  Fact 1.0: California State Polytechnic University, Pomona is also known as CPP, Cal Poly Pomona, or Cal Poly.
  Fact 1.1: California State Polytechnic University, Pomona is a public polytechnic university.
  Fact 1.2: California State Polytechnic University, Pomona is located in Pomona, California, United States.

Retrieved items:
<item 1>
Cal Poly Pomona College of Engineering
The Cal Poly Pomona College of Engineering is the engineering college at California State Polytechnic University, Pomona (Cal Poly Pomona or Cal Poly) located
in Pomona, California, United States.

 Well known for its "learn by doing" philosophy, the college's motto is: ""making imagination real"".

 Cal Poly has one of the top engineering college among public schools in California and, with over 5,600 students (as of fall 2015), it is also the largest
engineering college in Southern California, the second largest college of engineering in the California State University system, and the seventeenth largest
engineering college in the United States.

 In the 2017 "U.S. News & World Report" the College of Engineering is ranked the 5th best undergraduate program among public universities (11th overall) in the
West for Master's-granting universities, and "has one of the top ranked engineering programs, and graduates roughly 1 of every 14 engineers in the state of
California."
</item 1>

<item 2>
Cal Poly Pomona Broncos
The Cal Poly Pomona Broncos or Cal Poly Broncos are the athletic sports teams for the California State Polytechnic University, Pomona (Cal Poly Pomona).

 Cal Poly Pomona has 10 varsity sports teams and offers student participation in a wide range of sports including soccer, volleyball, track and field,
basketball, softball, and baseball.

 Cal Poly Pomona participates at the NCAA's Division II (DII) level in the California Collegiate Athletic Association (CCAA).

 The Broncos have 65 CCAA championships and 14 NCAA National Championships.

 Current and former Cal Poly athletes have won 7 Olympic medals (3 gold, 1 silver, and 3 bronze).
</item 2>

<item 3>
1981 Cal Poly Pomona Broncos football team
The 1981 Cal Poly Pomona Broncos football team represented California State Polytechnic University, Pomona during the 1981 NCAA Division II football season.

 Cal Poly Pomona competed in the California Collegiate Athletic Association (CCAA).
</item 3>

<item 4>
Julian A. McPhee
Julian Aeneas McPhee (February 7, 1896 – November 10, 1967) was the sixth university president of California Polytechnic State University, San Luis Obispo (Cal
Poly SLO) from 1933 to 1966 and the first president of California State Polytechnic University, Pomona (Cal Poly Pomona) from 1938 to 1966.
</item 4>

<item 5>
California State Polytechnic University, Pomona
California State Polytechnic University, Pomona (CPP, Cal Poly Pomona, or Cal Poly) is a public polytechnic university located in Pomona, California, United
States.

 It is one of two polytechnics in the California State University system.
</item 5>

<item 6>
Dalhousie University
Dalhousie University (commonly known as Dal) is a public research university in Nova Scotia, Canada, with three campuses in Halifax, a fourth in Bible Hill, and
medical teaching facilities in Saint John, New Brunswick.

 Dalhousie offers more than 4,000 courses and 180 degree programs in twelve undergraduate, graduate, and professional faculties.

 The university is a member of the U15, a group of research-intensive universities in Canada.
</item 6>

<item 7>
Cal Poly Pomona University Library
The Cal Poly Pomona University Library is the main library on the campus of California State Polytechnic University, Pomona (Cal Poly Pomona).

 It has in its collection 670,580 books; 6,883 serial subscriptions and 10,417 audiovisual materials.
</item 7>

<item 8>
International Polytechnic High School
International Polytechnic High School, commonly abbreviated merely as iPoly, is a public college preparatory demonstration high school (9-12) located on the
California State Polytechnic University, Pomona (Cal Poly Pomona) campus and operated by the Los Angeles County Office of Education (LACOE) in conjunction with
the College of Education and Integrative Studies at the university.

 iPoly's curriculum is fully accredited by the Western Association of Schools and Colleges and is approved by the University of California and California State
University.

 It maintains a unique affiliation with the Los Angeles County High School for the Arts (LACHSA), which is also run by LACOE.

 Since iPoly does not lie in a fixed school district, it draws students from throughout Los Angeles, San Bernardino, Orange and Riverside counties.

 The majority of students come from the Pomona and San Gabriel valleys.

 In 2009 and 2013, iPoly was honored as a California Distinguished School by the California Department of Education.
</item 8>

<item 9>
Cal Poly Pomona Broncos women's basketball
The Cal Poly Pomona Broncos women's basketball team is the women's basketball team that represents California State Polytechnic University, Pomona, in Pomona,
California.

 The school's team currently competes in the California Collegiate Athletic Association.
</item 9>

<item 10>
Cal Poly Pomona College of Environmental Design
The Cal Poly Pomona College of Environmental Design (CENV) is a college part of the California State Polytechnic University, Pomona (Cal Poly Pomona).

 The college houses over 1,600 students; making it one of largest environmental design programs in the United States.

 The college offers bachelor's degrees in five departments, as well as three master's degree programs.

 It is the only academic unit within the California State University system to be associated with a Pritzker Prize laureate (often referred to as "The Nobel
Prize in Architecture").
</item 10>
=== END ITEM id=f90353d54d51bc56 ===

=== ITEM id=6c8ff7eaa0afa02f ===
Gold sentences and facts:
Sentence 0: The Random House Tower, also known as the Park Imperial Apartments, is a 52-story mixed-use tower in New York City, United States, that is used as the headquarters of book publisher Random House and a luxury apartment complex.
  Fact 0.0: The Random House Tower is also known as the Park Imperial Apartments.
  Fact 0.1: The Random House Tower is a 52-story mixed-use tower.
  Fact 0.2: The Random House Tower is in New York City, United States.
  Fact 0.3: The Random House Tower is used as the headquarters of book publisher Random House.
  Fact 0.4: The Random House Tower is used as a luxury apartment complex.
Sentence 1:  It currently carries the Vornado Realty Trust corporate headquarters.
  Fact 1.0: The Random House Tower currently carries the Vornado Realty Trust corporate headquarters.

Retrieved items:
<item 1>
383 Madison Avenue
383 Madison Avenue is an office building owned and occupied by JP Morgan Chase in New York City on a full block bound by Madison Avenue and Vanderbilt Avenue
between East 46th and 47th Streets.

 Formerly known as the Bear Stearns Building, it housed the world headquarters of the now-defunct Bear Stearns from the building's completion until Bear's
collapse and sale to JPMorgan Chase in 2008.

 The building now houses the New York offices for J.P. Morgan's investment banking division, which formerly occupied 277 Park Avenue.

 Both 383 Madison and 277 Park are adjacent to JPMorgan Chase's world headquarters at 270 Park Avenue.
</item 1>

<item 2>
270 Park Avenue
270 Park Avenue (also known as the JPMorgan Chase Tower and formerly the Union Carbide Building) is a high-rise office building located in Midtown Manhattan,
New York City.

 It was designed by Natalie de Blois for Skidmore, Owings & Merrill.
</item 2>

<item 3>
245 Park Avenue
245 Park Avenue (formerly American Tobacco Company Building, American Brands Building and Bear Stearns Building) is a 648-ft (198 m) tall skyscraper in New York
City, New York.

 It was completed in 1967 and has 48 floors (1.7 million square feet).

 Shreve, Lamb and Harmon designed the building.

 The Building Owners and Managers Association awarded the 2000/2001 Pinnacle Award to 245 Park Avenue.

 The site used to be occupied by the second Grand Central Palace exhibition hall, which was demolished in 1964 to make way for 245 Park Avenue.
</item 3>

<item 4>
Random House Tower
The Random House Tower, also known as the Park Imperial Apartments, is a 52-story mixed-use tower in New York City, United States, that is used as the
headquarters of book publisher Random House and a luxury apartment complex.

 The book publisher entrance is on Broadway and goes up to 27 floors, while the apartment complex entrance is on West 56th Street.
</item 4>

<item 5>
888 7th Avenue
888 7th Avenue is a 628 ft (191m) tall modern-style office skyscraper in Midtown Manhattan which was completed in 1969 and has 46 floors.

 Emery Roth & Sons designed the building, which is tied with Central Park Place for the 65th tallest building in New York City.

 It currently carries the Vornado Realty Trust corporate headquarters.

 Previously known as the Arlen Building, its namesake being the company responsible for its construction, Arlen Realty & Development Corporation.

 The Red Eye Grill is located in the building at street level.
</item 5>

<item 6>
655 Park Avenue
655 Park Avenue is a Georgian-style co-op residential building on Manhattan's Upper East Side, located on Park Avenue between 67th Street and 68th Street,
adjacent to the Park Avenue Armory.

 It was developed in 1924 by Dwight P. Robinson & Company.

 The building at 655 Park Avenue was designed by architects James Edwin Ruthven Carpenter, Jr., often referred to by the initials "J.E.R. Carpenter", and Mott
B. Schmidt.

 Carpenter is considered the leading architect for luxury residential high-rise buildings in New York City in the early 1900s, while Schmidt is known for his
buildings in the American Georgian Classical style, including Sutton Place and houses for New York City's society figures and business elite.
</item 6>

<item 7>
90 West Street
90 West Street (alternatively West Street Building) is a building in Lower Manhattan, New York City.

 It was designed by architect Cass Gilbert and structural engineer Gunvald Aus for the West Street Improvement Corporation.

 When completed in 1907, the building's Gothic styling and ornamentation served to emphasize its 23-story height, and foreshadowed Gilbert's later work on the
Woolworth Building.

 Originally built as an office building, the main tenant was the Delaware, Lackawanna and Western Railroad and the top floor was occupied by Garret's
Restaurant, which advertised itself as the "world's highest restaurant".
</item 7>

<item 8>
Rubin Schron
Other buildings he owns include the Monterey, a 521-unit rental multifamily building on Manhattan's Upper East Side; over the decades, Schron has also amassed a
portfolio of Mitchell-Lama apartment buildings whose values have been skyrocketing to record values, after reverting to market rates when government subsidies
expired.

 In 2007, he sold nearly 4,000 units of former Mitchell-Lama properties in five complexes in Harlem and on Roosevelt Island for $940 million.

 Schron, who practices Orthodox Judaism, has eight children and 50 grandchildren.

 He and his family have lived in the same single-family home in Brooklyn for many decades.
</item 8>

<item 9>
200 West Street
200 West Street is the global headquarters of the Goldman Sachs investment banking firm.

 The building is a 749 ft , 44-story building located on West Street, between Vesey and Murray Streets in Lower Manhattan.

 It is adjacent to the World Financial Center and the Conrad Hotel, the Verizon Building to the east across West Street, and diagonally opposite the World Trade
Center.

 It is the only office building in Battery Park City north of the World Financial Center.
</item 9>

<item 10>
Transportation Building
The Transportation Building is a 44-story office building located at 225 Broadway on the corner of Barclay Street in the Civic Center neighborhood of Lower
Manhattan in New York City.

 It also carries the address 2-4 Barclay Street.

 It was built in 1927 and was designed by the architecture firm of York & Sawyer, in the Renaissance Revival style, using setbacks common to skyscrapers built
after the adoption of the 1916 Zoning Resolution.

 It sits across Barclay Street from the Woolworth Building.
</item 10>
=== END ITEM id=6c8ff7eaa0afa02f ===

=== ITEM id=75e2b448789959ac ===
Gold sentences and facts:
Sentence 0: Remember the Titans is a 2000 American sports film produced by Jerry Bruckheimer and directed by Boaz Yakin.
  Fact 0.0: Remember the Titans is a 2000 American sports film
  Fact 0.1: Remember the Titans was produced by Jerry Bruckheimer
  Fact 0.2: Remember the Titans was directed by Boaz Yakin
Sentence 1: My Favorite Martian is a 1999 American science fiction comedy film starring Christopher Lloyd, Jeff Daniels, Daryl Hannah, Elizabeth Hurley, Wallace Shawn and Ray Walston, based on the 1960s television series of the same name in which Walston starred.
  Fact 1.0: My Favorite Martian is a 1999 American science fiction comedy film
  Fact 1.1: My Favorite Martian stars Christopher Lloyd
  Fact 1.2: My Favorite Martian stars Jeff Daniels
  Fact 1.3: My Favorite Martian stars Daryl Hannah
  Fact 1.4: My Favorite Martian stars Elizabeth Hurley
  Fact 1.5: My Favorite Martian stars Wallace Shawn
  Fact 1.6: My Favorite Martian stars Ray Walston
  Fact 1.7: My Favorite Martian is based on the 1960s television series of the same name
  Fact 1.8: Ray Walston starred in the 1960s television series My Favorite Martian
Sentence 2:  The film grossed $36.8 million domestically against a budget of $65 million.
  Fact 2.0: The film (My Favorite Martian) grossed $36.8 million domestically
  Fact 2.1: The film (My Favorite Martian) had a budget of $65 million

Retrieved items:
<item 1>
Remember the Titans
Remember the Titans is a 2000 American sports film produced by Jerry Bruckheimer and directed by Boaz Yakin.

 The screenplay, written by Gregory Allen Howard, is based on the true story of African-American coach Herman Boone, portrayed by Denzel Washington, and his
attempt to integrate the T. C. Williams High School football team in Alexandria, Virginia, in 1971.

 Will Patton portrays Bill Yoast, Boone's assistant coach.

 Real-life athletes Gerry Bertier and Julius Campbell are portrayed by Ryan Hurst and Wood Harris, respectively.
</item 1>

<item 2>
Titan A.E.
Titan A.E. is a 2000 American animated science fiction film directed by Don Bluth and Gary Goldman.

 Its title refers to the spacecraft central to the plot, with "A.E." meaning "After Earth".

 It stars Matt Damon, Bill Pullman, John Leguizamo, Nathan Lane, Janeane Garofalo and Drew Barrymore.

 The film's animation technique combines traditional hand-drawn animation and extensive use of computer generated imagery.

 Its working title was "Planet Ice".

 It was theatrically released on June 16, 2000, by 20th Century Fox and was the final film for Fox Animation Studios.

 The film grossed $36.8 million on a $75–$90 million budget, making a $100-million loss for the studio.
</item 2>

<item 3>
The entries cover diverse films and careers: crime drama *Deadfall* and its *Arsenal* follow-up; *Rogue One*’s rebels stealing Death Star plans; Mel Gibson,
Nicolas Cage, James Garner and Tom Cruise filmographies; Everest disaster drama *Everest*; integration-themed sports films *Remember the Titans* and *McFarland,
USA*; acclaimed *True Grit* and *The Hurt Locker*; espionage film *The Ipcress File*; Vietnam
</item 3>

<item 4>
My Favorite Martian (film)
My Favorite Martian is a 1999 American science fiction comedy film starring Christopher Lloyd, Jeff Daniels, Daryl Hannah, Elizabeth Hurley, Wallace Shawn and
Ray Walston, based on the 1960s television series of the same name in which Walston starred.

 It was directed by Donald Petrie and written by Sherri Stoner and Deanna Oliver, based on the television series created by John L. Greene.

 Creatures were created by Amalgamated Dynamics from designs by Jordu Schell.

 The film grossed $36.8 million domestically against a budget of $65 million.
</item 4>

<item 5>
Tom Cruise filmography
Tom Cruise is an American actor and producer who made his film debut with a minor role in the 1981 romantic drama "Endless Love".

 Two years later he made his breakthrough by starring in the romantic comedy "Risky Business" (1983), which garnered Cruise his first nomination for the Golden
Globe Award for Best Actor – Motion Picture Musical or Comedy.

 In 1986, Cruise played a fighter pilot in the Tony Scott-directed action drama "Top Gun" (the highest-grossing film that year), and also starred opposite Paul
Newman in the Martin Scorsese-directed drama "The Color of Money".

 Two years later he played opposite Dustin Hoffman in the Academy Award for Best Picture-winning drama "Rain Man" (1988), and also appeared in the Golden
Raspberry Award for Worst Picture-winning romantic drama "Cocktail" (1988).

 In doing so Cruise became the first and only person as of 2014 to star in a Best Picture Oscar winner and a Worst Picture Razzie winner in the same year.
</item 5>

<item 6>
The entries cover films across action, comedy, romance, horror, and drama. Bruce Willis stars in titles including *Last Man Standing* (a credited *Yojimbo*
remake), *Unbreakable*, *Signs*, and *Die Hard with a Vengeance*. Shyamalan’s *Signs* explores faith and alien encounters; *Unbreakable* launches a superhero
trilogy. Other highlights include *The Whole Nine Yards*, *Striking Distance*, *Mercury Rising
</item 6>

<item 7>
The collection spans films and TV from 1947–2010: Bruce Willis’s dramas, action comedies and thrillers include *In Country* (Golden Globe-nominated), *Last Man
Standing* (a *Yojimbo* remake), *The Whole Nine Yards* and *Cop Out*. Other highlights: *The Bears and I*; teen comedies *Can’t Buy Me Love* and *The Night
Before*; *The Country Girl* (Oscar-winning screenplay
</item 7>

<item 8>
Mercury Rising
Mercury Rising is a 1998 American political action thriller film starring Bruce Willis and Alec Baldwin.

 Directed by Harold Becker, the movie is based on Ryne Douglas Pearson's 1996 novel originally published as "Simple Simon".

 Willis plays Art Jeffries, an undercover FBI agent who protects a 9-year-old boy with autism who is targeted by government assassins after he cracks a top
secret government code.
</item 8>

<item 9>
McFarland, USA
McFarland, USA (also known as McFarland) is a 2015 American sports drama film directed by Niki Caro, produced by Mark Ciardi and Gordon Gray, written by
Christopher Cleveland, Bettina Gilois and Grant Thompson with music composed by Antônio Pinto.

 The film was co-produced by Walt Disney Pictures and Mayhem Pictures.

 Based on the true story of a 1987 cross country team from a mainly Latino high school in McFarland, California, the film stars Kevin Costner as Jim White, the
school's coach, who leads the team to win a state championship.

 The film also stars Maria Bello and Morgan Saylor.
</item 9>

<item 10>
Million Dollar Arm
Million Dollar Arm is a 2014 American biographical sports drama film directed by Craig Gillespie and produced by Walt Disney Pictures from a screenplay written
by Thomas McCarthy.

 The film is based on the true story of baseball pitchers Rinku Singh and Dinesh Patel who were discovered by sports agent J. B. Bernstein after winning a
reality show competition.
</item 10>
=== END ITEM id=75e2b448789959ac ===

=== ITEM id=c78c9ed919e777dd ===
Gold sentences and facts:
Sentence 0:  It was the first election for Governor of Maine, taking place after Maine separated from Massachusetts and was recognized as a state on March 15, 1820.
  Fact 0.0: The election was the first election for Governor of Maine.
  Fact 0.1: The election took place after Maine separated from Massachusetts.
  Fact 0.2: Maine was recognized as a state on March 15, 1820.
Sentence 1:  Maine's separation from Massachusetts came as a result of The Missouri Compromise.
  Fact 1.0: Maine's separation from Massachusetts came as a result of The Missouri Compromise.
Sentence 2:  This election saw the virtually unanimous election of William King, the man most chiefly responsible for the push for Maine statehood.
  Fact 2.0: The election saw the virtually unanimous election of William King.
  Fact 2.1: William King was the man most chiefly responsible for the push for Maine statehood.
Sentence 3: William King (February 9, 1768June 17, 1852) was an American merchant, shipbuilder, army officer, and statesman from Bath, Maine.
  Fact 3.0: William King was born February 9, 1768.
  Fact 3.1: William King died June 17, 1852.
  Fact 3.2: William King was an American merchant.
  Fact 3.3: William King was an American shipbuilder.
  Fact 3.4: William King was an American army officer.
  Fact 3.5: William King was an American statesman.
  Fact 3.6: William King was from Bath, Maine.

Retrieved items:
<item 1>
Anti-Nebraska movement
The Anti-Nebraska movement was a political alignment in the United States formed in opposition to the Kansas-Nebraska Act of 1854 and to its repeal of the
Missouri Compromise provision forbidding slavery in U.S. territories north of latitude 36° 30' N.

 (At the time, the name "Nebraska" could loosely refer to areas west of the Missouri River.)
</item 1>

<item 2>
1st Virginia State Regiment
The 1st Virginia State Regiment was a regiment of Virginia Militia that fought during the American Revolutionary War.

 The regiment was authorized by the General Assembly of the Commonwealth of Virginia in December 1776 as a force of regular troops for the Commonwealth's
defense.

 In 1777, Virginia had difficulty meeting its quota for the regular line of the Continental Army.

 As a result, in July 1777 under the command of Colonel George Gibson, the regiment began a march North to temporarily join the Continental Army in the
Philadelphia Campaign.

 In January 1778, the Virginia General Assembly passed an act directing that the 1st Virginia State Regiment "now in Continental service, be continued in said
service instead of the Ninth Virginia Regiment, made prisoners by the enemy in the Battle of Germantown."

 The regiment camped at Valley Forge in the winter of 1777-78 and at Middlebrook in the winter of 1778-79 and participated in the Battle of Monmouth.

 The regiment remained in the service of the Continental Army until late 1779 when redeployed Virginia.
</item 2>

<item 3>
Henry Smith Lane
Henry Smith Lane (February 24, 1811 – June 19, 1881) was a United States Representative, Senator, and the 13th Governor of Indiana; he was by design the
shortest-serving Governor of Indiana, having made plans to resign the office should his party take control of the Indiana General Assembly and elect him to the
United States Senate.

 He held that office for only two days, and was known for his opposition to slavery.

 A Whig until the party collapsed, he supported compromise with the south.

 He became an early leader in the Republican Party starting in 1856 serving as the president of the first party convention, delivering its keynote address, and
was influential in the nomination of Abraham Lincoln.

 With the repeal of the Missouri Compromise, he became a full-fledged
</item 3>

<item 4>
William King (governor)
William King (February 9, 1768June 17, 1852) was an American merchant, shipbuilder, army officer, and statesman from Bath, Maine.

 A proponent of statehood for Maine, he became its first governor when it separated from Massachusetts in 1820.
</item 4>

<item 5>
Pottawatomie massacre
The Pottawatomie massacre occurred during the night of May 24 and the morning of May 25, 1856.

 In reaction to the sacking of Lawrence, Kansas by pro-slavery forces, John Brown and a band of abolitionist settlers—some of them members of the Pottawatomie
Rifles—killed five settlers north of Pottawatomie Creek in Franklin County, Kansas.

 This was one of the many bloody episodes in Kansas preceding the American Civil War, which came to be known collectively as Bleeding Kansas.

 Bleeding Kansas was largely brought about by the Missouri Compromise and the Kansas–Nebraska Act.
</item 5>

<item 6>
John Bowman (pioneer)
Col. John (Johannes) Bowman (17 December 1738 – May 4, 1784) was an 18th-century American pioneer, colonial militia officer and sheriff, the first appointed in
Lincoln County, Kentucky.

 In 1781 he also presided as a justice of the peace over the first county court held in Kentucky.

 The first county-lieutenant and military governor of Kentucky County during the American Revolutionary War, Col. Bowman also, served in the American
Revolution, many times, second in command to General George Rogers Clark, during the Illinois Campaign, which, at the time, doubled the size of the United
States.
</item 6>

<item 7>
Louisa Adams
Louisa Catherine Johnson Adams (February 12, 1775 – May 15, 1852), wife of John Quincy Adams, was First Lady of the United States from 1825 to 1829.

 The daughter of the American Consul in London, she was the first First Lady to be born outside the United States, or outside areas that were later to become
part of the United States, such as the Thirteen Colonies – a distinction that would not be replicated until 192 years later by Melania Trump.
</item 7>

<item 8>
Compromise of 1790
The Compromise of 1790 was a compromise between Alexander Hamilton on the one hand and Thomas Jefferson and James Madison whereby Hamilton won the decision for
the national government to take over and pay the state debts, while Jefferson and Madison obtained the national capital (District of Columbia) for the South.

 The compromise resolved the deadlock in Congress.

 Southerners were blocking the assumption of state debts by the treasury, thereby destroying the Hamiltonian program for building a fiscally strong nation
state.

 Northerners rejected the proposal, much desired by Virginians, to locate the permanent national capital on the Virginia-Maryland border.

 The compromise made possible the passage of the Residence and Funding (Assumption) Acts in July and August 1790.

 Historian Jacob Cooke says it is, "generally regarded as one of the most important bargains in American history, ranking just below the better known Missouri
Compromise and the Compromise of 1850."
</item 8>

<item 9>
Maine gubernatorial election, 1820
The 1820 Maine gubernatorial election took place on April 3, 1820.

 It was the first election for Governor of Maine, taking place after Maine separated from Massachusetts and was recognized as a state on March 15, 1820.

 Maine's separation from Massachusetts came as a result of The Missouri Compromise.

 This election saw the virtually unanimous election of William King, the man most chiefly responsible for the push for Maine statehood.

 He had no opponents.
</item 9>

<item 10>
Missouri Compromise
The Missouri Compromise is the title generally attached to the legislation passed by the 16th United States Congress on May 8, 1820.

 The measures provided for the admission of Maine as a state along with Missouri as a slave state, thus maintaining the balance of power between North and
South.

 As part of the compromise, slavery was prohibited North of the 36°30′ parallel, excluding Missouri.

 President James Monroe signed the legislation on April 6, 1820.
</item 10>
=== END ITEM id=c78c9ed919e777dd ===

=== ITEM id=2fbed09604f8438d ===
Gold sentences and facts:
Sentence 0: Dorian Gray is the main character of the novel "The Picture of Dorian Gray" (1890) by Oscar Wilde.
  Fact 0.0: Dorian Gray is the main character of the novel The Picture of Dorian Gray.
  Fact 0.1: The Picture of Dorian Gray was published in 1890.
  Fact 0.2: The Picture of Dorian Gray was written by Oscar Wilde.
Sentence 1: The Picture of Dorian Gray is a philosophical novel by Oscar Wilde, first published complete in the July 1890 issue of "Lippincott's Monthly Magazine".
  Fact 1.0: The Picture of Dorian Gray is a philosophical novel.
  Fact 1.1: The Picture of Dorian Gray was written by Oscar Wilde.
  Fact 1.2: The Picture of Dorian Gray was first published complete.
  Fact 1.3: The Picture of Dorian Gray was first published complete in the July 1890 issue of Lippincott's Monthly Magazine.
Sentence 2:  The magazine's editor feared the story was indecent, and without Wilde's knowledge, deleted roughly five hundred words before publication.
  Fact 2.0: The editor of Lippincott's Monthly Magazine feared the story was indecent.
  Fact 2.1: The editor of Lippincott's Monthly Magazine, without Wilde's knowledge, deleted roughly five hundred words before publication.

Retrieved items:
<item 1>
A Portrait of Dorian Gray
A Portrait of Dorian Gray (2005) is the fashion designer and photographer Karl Lagerfeld's rendition of Oscar Wilde's novel "The Picture of Dorian Gray" (1891)
in photography.

 The models, Larry Scott and Eva Herzigová, star as Mr. and Mrs. Dorian Gray.
</item 1>

<item 2>
Dorian Gray (1970 film)
Dorian Gray (Italian: "Il dio chiamato Dorian") aka "The Sins of Dorian Gray" is a 1970 movie adaptation of Oscar Wilde's novel "The Picture of Dorian Gray"
starring Helmut Berger.

 The Italian title translates as "A God Called Dorian".
</item 2>

<item 3>
Dorian Gray (disambiguation)
Dorian Gray is the main character of the novel "The Picture of Dorian Gray" (1890) by Oscar Wilde.
</item 3>

<item 4>
El retrato de Dorian Gray
The Picture of Dorian Gray ("El retrato de Dorian Gray"), is a Mexican telenovela, based on "The Picture of Dorian Gray", the novel written by Oscar Wilde.

 The main character is the handsome young man called Dorian Gray (played by late Enrique Álvarez Félix).

 This telenovela was made in 1969.
</item 4>

<item 5>
The Picture of Dorian Gray (1945 film)
The Picture of Dorian Gray is a 1945 American horror-drama film based on Oscar Wilde's 1890 novel of the same name.

 Released in March 1945 by Metro-Goldwyn-Mayer, the film is directed by Albert Lewin and stars George Sanders as Lord Henry Wotton and Hurd Hatfield as Dorian
Gray.

 Shot primarily in black-and-white, the film features four inserts in 3-strip Technicolor of Dorian's portrait as a special effect (the first two of his
portrait's original state, and the second two after a major period of degeneration).
</item 5>

<item 6>
The Picture of Dorian Gray (1917 film)
The Picture of Dorian Gray (German: Das Bildnis des Dorian Gray) is a 1917 German silent fantasy film directed by Richard Oswald and starring Bernd Aldor, Ernst
Pittschau, and Ernst Ludwig.

 The film is based on the novel "The Picture of Dorian Gray" by Oscar Wilde.
</item 6>

<item 7>
Gate of Ivory, Gate of Horn
Gate of Ivory, Gate of Horn is a fantasy novel by British author Robert Holdstock.

 It was originally published in the United States in 1997 (and in the United Kingdom under the title "Gate of Ivory" in 1998.)

 The story is a prequel to "Mythago Wood" and explores Christian Huxley's quest into Ryhope Wood and the apparent suicide of his mother, Jennifer Huxley.

 The title of the book refers to the gates of horn and ivory described in both Homer's "Odyssey" and Virgil's "Aeneid".
</item 7>

<item 8>
The Picture of Dorian Gray
The Picture of Dorian Gray is a philosophical novel by Oscar Wilde, first published complete in the July 1890 issue of "Lippincott's Monthly Magazine".

 The magazine's editor feared the story was indecent, and without Wilde's knowledge, deleted roughly five hundred words before publication.

 Despite that censorship, "The Picture of Dorian Gray" offended the moral sensibilities of British book reviewers, some of whom said that Oscar Wilde merited
prosecution for violating the laws guarding the public morality.

 In response, Wilde aggressively defended his novel and art in correspondence with the British press, although he personally made excisions of some of the most
controversial material when revising and lengthening the story for book publication the following year.
</item 8>

<item 9>
The Picture of Dorian Gray (1910 film)
The Picture of Dorian Gray (1910), a.k.a. Dorian Grays Portræt, is a Danish silent film based on the novel "The Picture of Dorian Gray" by Oscar Wilde.
</item 9>

<item 10>
Dorian Gray (character)
Dorian Gray is the main character in Oscar Wilde's novel The Picture of Dorian Gray.

 Dorian Gray is a handsome, susceptible, wealthy young Victorian gentleman.

 He falls in love with and devotes himself to Sibyl Vane.

 Dorian has two friends, Basil and Henry.

 Basil is an artist who draws a portrait of Dorian that emphasizes his youthful, noble, and natural beauty.

 Henry is a friend who has a cynical nature and who shares various thoughts with Dorian.
</item 10>
=== END ITEM id=2fbed09604f8438d ===

=== ITEM id=3513b992bf63460d ===
Gold sentences and facts:
Sentence 0: Giovanni "Tinto" Brass (born 26 March 1933) is an Italian filmmaker.
  Fact 0.0: Giovanni 'Tinto' Brass was born on 26 March 1933.
  Fact 0.1: Giovanni 'Tinto' Brass is an Italian filmmaker.
Sentence 1: Baltasar Kormákur Samper (born 27 February 1966) is an Icelandic actor, theater and film director, and film producer.
  Fact 1.0: Baltasar Kormakur Samper was born on 27 February 1966.
  Fact 1.1: Baltasar Kormakur Samper is an Icelandic actor.
  Fact 1.2: Baltasar Kormakur Samper is an Icelandic theater director.
  Fact 1.3: Baltasar Kormakur Samper is an Icelandic film director.
  Fact 1.4: Baltasar Kormakur Samper is an Icelandic film producer.

Retrieved items:
<item 1>
Fermo posta Tinto Brass
Fermo posta Tinto Brass is a 1995 Italian comedy film-erotic film directed by Tinto Brass and set in vignettes.
</item 1>

<item 2>
Contraband (2012 film)
Contraband is a 2012 American crime thriller film directed by Baltasar Kormákur, starring Mark Wahlberg, Kate Beckinsale, Ben Foster, Caleb Landry Jones,
Giovanni Ribisi, Lukas Haas, Diego Luna and J. K. Simmons.

 The film is a remake of the 2008 Icelandic film "Reykjavík-Rotterdam" which Baltasar Kormákur starred in.

 It was released on January 13, 2012 in the United States by Universal Pictures.
</item 2>

<item 3>
Tinto Brass
Giovanni "Tinto" Brass (born 26 March 1933) is an Italian filmmaker.

 In the 1960s and 1970s, he directed many critically acclaimed avant-garde films of various genres.

 Today, he is mainly known for his later work in the erotic genre, with films such as "Caligula", "Così fan tutte" (released under the English title "All Ladies
Do It"), "Paprika", "Monella" ("Frivolous Lola") and "Trasgredire".
</item 3>

<item 4>
Graph query row:
person: Baltasar Kormákur
nationality: Icelandic
relationship: None
country_or_origin: None
</item 4>

<item 5>
Baltasar Kormákur
Baltasar Kormákur Samper (born 27 February 1966) is an Icelandic actor, theater and film director, and film producer.

 He is best known for directing the films "101 Reykjavík", "Hafið", "A Little Trip to Heaven" (starring Julia Stiles and Forest Whitaker), a film based on the
book "Mýrin" "(Jar City)" by Arnaldur Indriðason, "Contraband", "2 Guns" (starring Mark Wahlberg and Denzel Washington) and "Everest".

 His father is the Spanish painter Baltasar Samper and actor Baltasar Breki Samper is his son.
</item 5>

<item 6>
Trasgredire
Trasgredire (Transgressing or Cheeky) is a 2000 sex comedy directed by Tinto Brass, with Yuliya Mayarchuk in the lead role.

 Certain parallelisms are drawn between "Nerosubianco" (1969), another Tinto Brass film set in London.
</item 6>

<item 7>
Graph query row:
person: Tinto Brass
nationality: Italian
relationship: None
country_or_origin: None
</item 7>

<item 8>
Dropout (1970 film)
Dropout is a 1970 Italian romantic drama directed by Tinto Brass.

 It stars real-life couple, Franco Nero and Vanessa Redgrave.

 They also worked with Brass a year later on the drama, "La vacanza".

 "Dropout" was released in France on December 18, 1970, followed by a theatrical release in Italy on February 22, 1971.
</item 8>

<item 9>
John Steiner
He found himself in demand in Italy and moved there, appearing in a great number of Italian exploitation and B-films including police actioners ("Violent
Rome"), westerns ("Mannaja"), war films ("The Last Hunter"), nazisploitation ("Deported Women of the SS Special Section"), sci-fi adventure films ("Yor, the
Hunter from the Future"), and horror films, such as Mario Bava's "Shock", Dario Argento's "Tenebrae", and Ruggero Deodato's "Body Count".

 He also became a favourite of famed Italian filmmaker Tinto Brass, featuring in "Salon Kitty", the infamous "Caligula", "Action", and "Paprika".
</item 9>

<item 10>
Baltasar Breki Samper
Baltasar Breki Samper (born 22 July 1989), sometimes referred to as Baltasar Breki Baltasarsson, is an Icelandic actor.

 He is the son of Icelandic director Baltasar Kormákur.

 Before graduating from the Icelandic Academy of the Arts in 2015 Samper helped his father on several films as second assistant director.

 He is known for his role as Hjörtur in the Icelandic TV series "Trapped".
</item 10>
=== END ITEM id=3513b992bf63460d ===
