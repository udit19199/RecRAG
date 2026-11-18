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

=== ITEM id=2965aa1a060b5eae ===
Gold sentences and facts:
Sentence 0: Maximillian Oppenheimer (6 May 1902 – 26 March 1957), known as Max Ophüls (] ), was a German-born film director who worked in Germany (1931–1933), France (1933–1940 and 1950–1957), and the United States (1947–1950).
  Fact 0.0: Maximillian Oppenheimer was born on 6 May 1902.
  Fact 0.1: Maximillian Oppenheimer died on 26 March 1957.
  Fact 0.2: Maximillian Oppenheimer was known as Max Ophüls.
  Fact 0.3: Max Ophüls was a German-born film director.
  Fact 0.4: Max Ophüls worked in Germany from 1931 to 1933.
  Fact 0.5: Max Ophüls worked in France from 1933 to 1940.
  Fact 0.6: Max Ophüls worked in France from 1950 to 1957.
  Fact 0.7: Max Ophüls worked in the United States from 1947 to 1950.
Sentence 1: Shirley Clarke (October 2, 1919 – September 23, 1997) was an American experimental and independent filmmaker.
  Fact 1.0: Shirley Clarke was born on October 2, 1919.
  Fact 1.1: Shirley Clarke died on September 23, 1997.
  Fact 1.2: Shirley Clarke was an American experimental filmmaker.
  Fact 1.3: Shirley Clarke was an American independent filmmaker.

Retrieved items:
<item 1>
Oswalt Kolle
Oswalt Kolle (2 October 1928, Kiel – 24 September 2010, Amsterdam) was a German sex educator, who became famous during the late 1960s and early 1970s for his
numerous pioneering books and films on human sexuality.

 His work was translated into all major languages, while his films found an audience of 140 million worldwide.

 In his 1997 book "Open to Both Sides" he came out as bisexual.

 He was awarded the Magnus Hirschfeld Medal in 2000.
</item 1>

<item 2>
Jacques Natanson
French writer Jacques Natanson (15 May 1901 – 19 May 1975) first became involved in the movies in 1929 when one of his plays was adapted for the screen.

 He enjoyed a fruitful collaboration with Max Ophüls, on such films as "La Ronde" (1951, earning an Academy Award nomination), "Le Plaisir" (1952) and "Lola
Montès" (1955).
</item 2>

<item 3>
Lev Kuleshov
Lev Vladimirovich Kuleshov (Russian: Лев Влади́мирович Кулешо́в ; 13 January [O.S. 1 January] 1899 – 29 March 1970) was a Russian and Soviet filmmaker and film
theorist, one of the founders of the world's first film school, the Moscow Film School.

 He was given the title People's Artist of the RSFSR in 1969.

 He was intimately involved in development of the style of film making known as Soviet montage, especially it's psychological underpinning, including the use of
editing and the cut to emotionally influence the audience, a principle known as the Kuleshov effect.

 He also developed the theory of creative geography, which is the use of the action around a cut to connect otherwise disparate settings into a cohesive
narrative.
</item 3>

<item 4>
Max Ophüls
Maximillian Oppenheimer (6 May 1902 – 26 March 1957), known as Max Ophüls (] ), was a German-born film director who worked in Germany (1931–1933), France
(1933–1940 and 1950–1957), and the United States (1947–1950).

 He made nearly 30 films, the latter ones being especially notable: "La Ronde" (1950), "Le Plaisir" (1952), "The Earrings of Madame de..." (1953) and "Lola
Montès" (1955).
</item 4>

<item 5>
Dan Ouellette
Dan Ouellette is an American artist, illustrator, director, author and production designer.

 He began drawing as a child inspired by Escher, Dalí and Moebius.

 His drawings usually begin with very rough sketches and are intended to capture a moment of tension or eroticism.

 His art pushes his audience into areas where people feel less than comfortable, stating that the "US which is so puritanical, our bodies are usually a great
source of anxiety and fear".

 As a production designer, he has designed over fifteen independent feature films in the past decade as well as numerous commercials and music videos.

 He is best known as the director for the music videos "Blue" and "Looking Glass" by The Birthday Massacre, receiving over 4 million combined views on YouTube,
and has directed music videos for the industrial band Android Lust.

 He has done production design for many films over the years including "Chasing Sleep" starring Jeff Daniels.

 Dan's artwork is strongly themed around surrealism and mostly done in pencil.

 As an artist he has exhibited widely.

 He has been published in numerous anthology art books including "Bio-Mannerism" which also features work by H.R.
</item 5>

<item 6>
Shirley Clarke
Shirley Clarke (October 2, 1919 – September 23, 1997) was an American experimental and independent filmmaker.

 She was also a director and editor to many famous works.
</item 6>

<item 7>
Elmer Clifton
Elmer Clifton (March 14, 1890 – October 15, 1949) was an American writer, director and actor from the early silent days.

 A collaborator of D.W. Griffith, he appeared in "The Birth of a Nation" (1915) and "Intolerance" (1916) before giving up acting in 1917 to concentrate on work
behind the camera, with Griffith and Joseph Henabery as his mentors.

 His first feature-length solo effort as a director was "The Flame of Youth" with Jack Mulhall.
</item 7>

<item 8>
Lois Weber
Lois Weber (June 13, 1879 – November 13, 1939) was an American silent film actress, screenwriter, producer, and director, who is considered "the most important
female director the American film industry has known", and "one of the most important and prolific film directors in the era of silent films".

 Film historian Anthony Slide asserts that: "Along with D.W. Griffith, Weber was the American cinema's first genuine auteur, a filmmaker involved in all aspects
of production and one who utilized the motion picture to put across her own ideas and philosophies."
</item 8>

<item 9>
The Earrings of Madame de…
The Earrings of Madame de… (French: Madame de… ] ) is a 1953 drama film directed by Max Ophüls, adapted from Louise Leveque de Vilmorin's period novel by
Ophüls, Marcel Archard and Annette Wadement.

 The film is considered a masterpiece of the 1950s French cinema.

 Andrew Sarris called it "the most perfect film ever made".

 Ophüls said the story's construction attracted him, stating "there is always the same axis around which the action continually turns like a carousel.

 A tiny, scarcely visible axis: a pair of earrings."
</item 9>

<item 10>
Marcel Ophüls
Marcel Ophuls (] ; born 1 November 1927) is a documentary film maker and former actor, best known for his films "The Sorrow and the Pity" and "".
</item 10>
=== END ITEM id=2965aa1a060b5eae ===

=== ITEM id=6cb9ea6401908c98 ===
Gold sentences and facts:
Sentence 0: Wheelock Whitney may refer to one of three members of the Whitney family:
  Fact 0.0: Wheelock Whitney may refer to one of three members of the Whitney family.
Sentence 1: The Whitney family is an American family notable for their social prominence, wealth, business enterprises and philanthropy, founded by John Whitney (1592–1673) who came from London, England to Watertown, Massachusetts in 1635.
  Fact 1.0: The Whitney family is an American family.
  Fact 1.1: The Whitney family is notable for their social prominence, wealth, business enterprises and philanthropy.
  Fact 1.2: The Whitney family was founded by John Whitney (1592–1673).
  Fact 1.3: John Whitney came from London, England.
  Fact 1.4: John Whitney came to Watertown, Massachusetts.
  Fact 1.5: John Whitney came to Watertown, Massachusetts in 1635.

Retrieved items:
<item 1>
Gertrude Vanderbilt Whitney
Gertrude Vanderbilt Whitney (January 9, 1875 – April 18, 1942) was an American sculptor, art patron and collector, and founder in 1931 of the Whitney Museum of
American Art in New York City.

 She was a prominent social figure and hostess, who was born into the wealthy Vanderbilt family and married into the Whitney family.
</item 1>

<item 2>
John Hay Whitney
John Hay Whitney (August 17, 1904 – February 8, 1982), colloquially known as Jock Whitney, was U.S. Ambassador to the United Kingdom, publisher of the "New York
Herald Tribune", and president of the Museum of Modern Art.

 He was a member of the Whitney family.
</item 2>

<item 3>
Greentree Stable
Greentree Stable, in Red Bank, New Jersey, was a major American thoroughbred horse racing stable and breeding farm established in 1914 by Payne Whitney of the
Whitney family of New York City.

 Payne Whitney operated a horse farm and stable at Saratoga Springs, New York with his brother Harry Payne Whitney, who also had a large stable of horses.

 Greentree Stable had a training base at Aiken, South Carolina, while Greentree Farm in Lexington, Kentucky was established in 1925 as its breeding arm.
</item 3>

<item 4>
Rockefeller family
The Rockefeller family ( ) is an American industrial, political, and banking family that made one of the world's largest fortunes in the oil business during the
late 19th and early 20th centuries, with John D. Rockefeller and his brother William Rockefeller primarily through Standard Oil.

 The family is also known for its long association with and control of Chase Manhattan Bank.

 They are considered to be one of the most powerful families, if not the most powerful family, in the history of the United States.
</item 4>

<item 5>
James Scollay Whitney
James Scollay Whitney (May 19, 1811 – October 24, 1878) was an American business executive and politician.

 He was the father of Henry Melville Whitney and William Collins Whitney, founders of the Whitney family business interests.
</item 5>

<item 6>
Charles Andrew Whitney
Charles Andrew Whitney (November 14, 1864- December 31, 1912) was an American businessman and industrialist in the late 19th century, born in Princeton,
Massachusetts.

 He was part of the prominent American Whitney family.

 In 1859, Charles, his brother Levi L. Whitney, and Orville E. Thompson helped lead the large-scale manufacturing of leather boots and shoes in Chicago and were
attributed with successfully running the first factory of its kind there.

 Whitney himself held patents for the manufacture of leather.
</item 6>

<item 7>
Wheelock Whitney
Wheelock Whitney may refer to one of three members of the Whitney family:
</item 7>

<item 8>
Whitney family
The Whitney family is an American family notable for their social prominence, wealth, business enterprises and philanthropy, founded by John Whitney (1592–1673)
who came from London, England to Watertown, Massachusetts in 1635.

 The historic family mansion in Watertown, known as The Elms, was built in 1710.
</item 8>

<item 9>
Harry Payne Whitney
Harry Payne Whitney (April 29, 1872 – October 26, 1930) was an American businessman, thoroughbred horse breeder, and member of the prominent Whitney family.
</item 9>

<item 10>
Ross Winans
Ross Winans (1796–1877) was an American inventor, mechanic, and builder of locomotives and railroad machinery.

 He is also noted for design of pioneering cigar-hulled ships.

 Winans, one of the United States' first multi-millionaires, was involved in national and state politics, a southern-sympathizer and was a vehement "states'
rights" advocate.

 His outspoken anti-federal stance as a member of the Maryland House of Delegates, the lower chamber of the General Assembly, (state legislature) led to his
temporary arrest on board a Baltimore and Ohio Railroad train returning from an early session of the legislature held in the western Maryland town of Frederick
to avoid the Union Army-occupied state capital of Annapolis in April–May, 1861, to consider the possibilities of state secession, during the early decisive
period of the American Civil War.

 Winans was related to James McNeill Whistler through marriage (Whistler's brother George married Winans' daughter Julia).
</item 10>
=== END ITEM id=6cb9ea6401908c98 ===

=== ITEM id=e8c5f7e921860f2e ===
Gold sentences and facts:
Sentence 0: Big Fish is a musical with music and lyrics by Andrew Lippa and a book by John August.
  Fact 0.0: Big Fish is a musical.
  Fact 0.1: Big Fish has music by Andrew Lippa.
  Fact 0.2: Big Fish has lyrics by Andrew Lippa.
  Fact 0.3: Big Fish has a book by John August.
Sentence 1:  He is a resident artist at the Ars Nova Theater in New York City.
  Fact 1.0: He is a resident artist at the Ars Nova Theater.
  Fact 1.1: The Ars Nova Theater is in New York City.

Retrieved items:
<item 1>
Marc Blitzstein
Marcus Samuel Blitzstein (March 2, 1905January 22, 1964), was an American composer, lyricist, and librettist.

 He won national attention in 1937 when his pro-union musical "The Cradle Will Rock", directed by Orson Welles, was shut down by the Works Progress
Administration.

 He is known for "The Cradle Will Rock" and for his Off-Broadway translation/adaptation of "The Threepenny Opera" by Bertolt Brecht and Kurt Weill.

 His works also include the opera "Regina", an adaptation of Lillian Hellman's play "The Little Foxes"; the Broadway musical "Juno", based on Seán O'Casey's
play "Juno and the Paycock"; and "No for an Answer".

 He completed translation/adaptations of Brecht's and Weill's musical play "Rise and Fall of the City of Mahagonny" and of Brecht's play "Mother Courage and Her
Children" with music by Paul Dessau.
</item 1>

<item 2>
Aaron Barrett
Aaron Asher Barrett (born August 30, 1974) is the lead singer, lead guitarist, and primary songwriter of the American ska-punk band Reel Big Fish.

 Prior to forming Reel Big Fish, Barrett played trombone in another ska-punk act, The Scholars, along with future Reel Big Fish bandmates Scott Klopfenstein and
Grant Barry.
</item 2>

<item 3>
Maurice Browne
Maurice Browne (12 February 1881 – 21 January 1955), born in Reading, England, was best known as a theater producer in the United States and the UK.

 The Cambridge-educated Browne was also a poet, actor, and theater director.

 He has been credited, along with his then-wife Ellen Van Volkenburg, with being the founder of the Little Theatre Movement in America through his work with the
Chicago Little Theatre.

 Browne and Van Volkenburg went on to found the department of drama at the Cornish School in Seattle in 1918, now Cornish College of the Arts.

 Browne's greatest triumph came in 1929 when he produced "Journey's End", by R. C. Sherriff in London.
</item 3>

<item 4>
Lin-Manuel Miranda
Lin-Manuel Miranda ( ; born January 16, 1980) is an American composer, lyricist, playwright, and actor best known for creating and starring in the Broadway
musicals "Hamilton" and "In the Heights".

 He co-wrote the songs for Disney's "Moana" soundtrack (2016) and is set to star in their upcoming film "Mary Poppins Returns".

 Miranda's awards include a Pulitzer Prize, two Grammy Awards, an Emmy Award, a MacArthur Fellowship, and three Tony Awards.
</item 4>

<item 5>
Big Fish (musical)
Big Fish is a musical with music and lyrics by Andrew Lippa and a book by John August.

 It is based on Daniel Wallace’s 1998 novel, "", and the 2003 film "Big Fish" written by August and directed by Tim Burton.
</item 5>

<item 6>
Love Birds (musical)
Love Birds is an original musical with book, music and lyrics by Robert J. Sherman.

 The musical officially premiered at the 2015 Edinburgh Fringe Festival at The Pleasance on August 7, 2015.

 The production was directed and choreographed by Stewart Nicholls.

 Prior to the Edinburgh run, there were two previews held in London at the Lost Theatre (in Stockwell) on July 28 and 29.

 The Original Edinburgh Cast Recording was recorded on July 30 and mixed and mastered on July 31 and August 3 and was first released by SimG Records on August
12 at the Edinburgh Festival.

 Two Edinburgh previews took place on August 5 and 6 with Press Night taking place on August 8.

 The final festival performance took place on August 31.

 The show starred Ruth Betteridge, Greg Castiglioni, John Guerrasio, George Knapper, Jonny Purchase, Joanna Sawyer, Anna Stolli, Rafe Watts and Ryan Willis.
</item 6>

<item 7>
Marc Blitzstein
Blitzstein also composed music for films, such as "Surf and Seaweed" (1931) and "The Spanish Earth" (1937), and he contributed two songs to the original 1960
production of Hellman's play "Toys in the Attic".
</item 7>

<item 8>
Love Birds (musical)
In its end of the year round-up of regional British fringe musical theatre, "Musical Theatre Review" hailed "Love Birds": "a triumph of whimsy, a children’s
story written with adult sensibilities and featuring a cracking score that celebrated the lost era of vaudeville.

 Blessed with an equally strong design from Gabriella Slade and some marvellously sensitive performances, Love Birds was probably the most accomplished piece of
musical theatre to arise from the many delights at this year’s festival."
</item 8>

<item 9>
Andrew Lippa
Andrew Lippa (December 22, 1964) is an American composer, lyricist, book writer, performer, and producer.

 He is a resident artist at the Ars Nova Theater in New York City.
</item 9>

<item 10>
Marisha Chamberlin
Marisha Anne Chamberlain (born January 6, 1952) is an American writer.

 Her most recent work, the libretto for "Mortals & Angels", a collaboration with American composer Carol Barnett, premiered at Carnegie Hall in 2016, a
companion piece to their widely produced collaborative piece, "The World Beloved: A Bluegrass Mass".

 Her stage plays, both original works and adaptations are widely produced.

 Her debut novel, "The Rose Variations", was published by Soho Press in 2009.

 Her play, "Scheherazade", won the Dramatists Guild/CBS National Award, and in her screenplay version, played on public television across the country, and was
screened at the British Film Institute Festival in the category of Best of American Public Television.
</item 10>
=== END ITEM id=e8c5f7e921860f2e ===

=== ITEM id=d59d9ebdb57cef7b ===
Gold sentences and facts:
Sentence 0:  Launched in 2006 and backed by Canadian industrialist David Harrison Gilmour (the founder of FIJI Water LLC), the bimonthly magazine is distributed by Zinio Systems, Inc.
  Fact 0.0: The magazine was launched in 2006.
  Fact 0.1: The magazine is backed by Canadian industrialist David Harrison Gilmour.
  Fact 0.2: David Harrison Gilmour is the founder of FIJI Water LLC.
  Fact 0.3: The magazine is bimonthly.
  Fact 0.4: The magazine is distributed by Zinio Systems, Inc.
Sentence 1:  The first issue appeared in January-February 2007.
  Fact 1.0: The first issue of the magazine appeared in January-February 2007.
Sentence 2: La Belle Assemblée (in full La Belle Assemblée or, Bell's Court and Fashionable Magazine Addressed Particularly to the Ladies) was a British women's magazine published from 1806 to 1837, founded by John Bell (1745–1831).
  Fact 2.0: La Belle Assemblée is also known in full as 'La Belle Assemblée or, Bell's Court and Fashionable Magazine Addressed Particularly to the Ladies'.
  Fact 2.1: La Belle Assemblée was a British women's magazine.
  Fact 2.2: La Belle Assemblée was published from 1806 to 1837.
  Fact 2.3: La Belle Assemblée was founded by John Bell.
  Fact 2.4: John Bell lived from 1745 to 1831.

Retrieved items:
<item 1>
SexIs
SexIs Magazine is a quarterly print publication and daily webzine devoted to sex and sexual culture, founded in 2008 by Web Merchants, parent company of sex toy
e-tailer EdenFantasys.com.

 The first print issue debuted in November 2009, distributed nationwide as an insert in "BUST magazine".

 The website publishes articles, columns, video presentations and news items daily.
</item 1>

<item 2>
VIVmag
VIVmag is one of the earliest exclusively digital, interactive women's lifestyle magazines.

 Launched in 2006 and backed by Canadian industrialist David Harrison Gilmour (the founder of FIJI Water LLC), the bimonthly magazine is distributed by Zinio
Systems, Inc.

 The first issue appeared in January-February 2007.

 There is no print version.

 The paperless title can be viewed in most browsers or downloaded to your computer and read via the Zinio Reader.

 The magazine is based in Thousand Oaks, California.
</item 2>

<item 3>
Essence (magazine)
Essence is a monthly magazine for African American women between the ages of 18 and 49.

 It is the only magazine that focuses on reaching an audience of black women, revolves around the black woman experience, and has remained for a long period of
time.

 The magazine covers fashion, lifestyle and beauty, with an intimate girlfriend-to-girlfriend tone, and its slogan "Fierce, Fun, and Fabulous" suggests the
magazine's goal of empowering African-American women.

 The topics the magazine discusses range from celebrities, to fashion, to point-of-view pieces addressing current issues in the African-American community.

 A number of its readers engage closely and personally with the publication, and it claims to be the magazine "for and about Black women".
</item 3>

<item 4>
Annales de chimie et de physique
Annales de chimie et de physique (French for "Annals of Chemistry and of Physics") is a scientific journal that was founded in Paris, France, in 1789 under the
title "Annales de chimie".

 One of the early editors was the French chemist Antoine Lavoisier.

 In 1815, it became the "Annales de chimie et de physique", and was published under that name for the next 100 years.
</item 4>

<item 5>
Diva (magazine)
DIVA is a leading lesbian magazine in the United Kingdom, published monthly.

 It was launched in March 1994 by Millivres Ltd, under the editorship of Frances Williams and is now published by Twin Media Group, which is owned by Silke
Bader and Linda Riley.

 Silke Bader is also the publisher of Curve magazine in the USA and LOTL in Australia, and Linda Riley runs the European Diversity Awards and is the former
publisher of g3 magazine in the UK.
</item 5>

<item 6>
La Belle (ship)
La Belle was one of Robert de La Salle's four ships when he explored the Gulf of Mexico with the ill-fated mission of starting a French colony at the mouth of
the Mississippi River in 1685.

 "La Belle" was wrecked in present-day Matagorda Bay the following year, dooming La Salle's Texas colony to failure.

 For over three centuries the wreckage of "La Belle" lay forgotten until it was discovered by a team of state archaeologists in 1995.

 The discovery of La Salle's flagship was regarded as one of the most important archaeological finds of the century in Texas, and a major excavation was
launched by the state of Texas that, over a period of about a year, recovered the entire shipwreck and over a million artifacts.
</item 6>

<item 7>
La Belle Assemblée
La Belle Assemblée (in full La Belle Assemblée or, Bell's Court and Fashionable Magazine Addressed Particularly to the Ladies) was a British women's magazine
published from 1806 to 1837, founded by John Bell (1745–1831).
</item 7>

<item 8>
Auguste Brancart
August or Auguste Brancart (21 July 1851 - 1894?)

 was a Belgian publisher of pornographic literature, credited with the first publication of "My Secret Life".

 He published translations of English pornography into French and vice versa for English publishers such as Edward Avery.

 He also published work of the Decadent movement such as "Monsieur Vénus" by Rachilde.
</item 8>

<item 9>
André Laguerre
Marc André Laguerre (February 21, 1915 – January 18, 1979) was a journalist and magazine editor, best known as the managing editor of "Sports Illustrated" from
1960 to 1974, during which time he oversaw the growth in the magazine from a niche publication to become the industry leader in weekly sports magazines.

 It was under his leadership that the annual "Sports Illustrated Swimsuit Issue" was first published.

 When he retired in 1974, he had been managing editor of the magazine for 704 issues, then a record among magazines published by Time, Inc., SI's parent
company.
</item 9>

<item 10>
Elle (magazine)
Elle is a worldwide lifestyle magazine of French origin that focuses on fashion, beauty, health, and entertainment.

 "Elle" is also the world's best-selling fashion magazine.

 It was founded by Pierre Lazareff and his wife Hélène Gordon in 1945.

 The title, in French, means "she" or "her".
</item 10>
=== END ITEM id=d59d9ebdb57cef7b ===

=== ITEM id=816dd144b2f11656 ===
Gold sentences and facts:
Sentence 0: Ciaran Gribbin (born 1976) is a Northern Irish singer-songwriter, guitarist and producer who hails from Castledawson, Northern Ireland.
  Fact 0.0: Ciaran Gribbin was born in 1976.
  Fact 0.1: Ciaran Gribbin is a Northern Irish singer-songwriter.
  Fact 0.2: Ciaran Gribbin is a guitarist.
  Fact 0.3: Ciaran Gribbin is a producer.
  Fact 0.4: Ciaran Gribbin hails from Castledawson, Northern Ireland.
Sentence 1: Brett Allen Scallions (born December 21, 1971) is an American musician.
  Fact 1.0: Brett Allen Scallions was born on December 21, 1971.
  Fact 1.1: Brett Allen Scallions is an American musician.
Sentence 2:  He is the lead vocalist, rhythm guitarist, and lyricist of post-grunge band Fuel.
  Fact 2.0: Brett Allen Scallions is the lead vocalist of the post-grunge band Fuel.
  Fact 2.1: Brett Allen Scallions is the rhythm guitarist of the post-grunge band Fuel.
  Fact 2.2: Brett Allen Scallions is the lyricist of the post-grunge band Fuel.

Retrieved items:
<item 1>
Brett Scallions
Brett Allen Scallions (born December 21, 1971) is an American musician.

 He is the lead vocalist, rhythm guitarist, and lyricist of post-grunge band Fuel.
</item 1>

<item 2>
Puppet Strings
Puppet Strings is the fifth studio album by American rock band Fuel.

 Released March 4, 2014, it is their first studio effort since 2003's "Natural Selection" to feature original lead singer Brett Scallions and the first Fuel
album not to feature original songwriter/guitarist, Carl Bell and bassist Jeff Abercrombie.

 Additionally, no one who participated in the recording of Fuel's previous album "Angels & Devils" participated in the recording of "Puppet Strings".
</item 2>

<item 3>
Yeah! (Fuel song)
"Yeah!"

 is the first song released from Fuel's album "Puppet Strings".

 It is also the first song released featuring original lead singer Brett Scallions since Fuel's 2003 studio album "Natural Selection".
</item 3>

<item 4>
Graph query row:
person: Brett Scallions
relationship: MEMBER_OF
band: Fuel
role: None
period: None
status: None
</item 4>

<item 5>
Grant Nicholas
Grantley Jonathan "Grant" Nicholas (born 12 November 1967) is a Welsh musician, best known as the lead singer and lead guitarist of the rock band Feeder.
</item 5>

<item 6>
Soul to Preach To
"Soul to Preach To" is the first single released from Fuel's album "Puppet Strings".

 It is also the first single released featuring original lead singer Brett Scallions since Fuel's 2003 studio album "Natural Selection".
</item 6>

<item 7>
Ciaran Gribbin
Ciaran Gribbin (born 1976) is a Northern Irish singer-songwriter, guitarist and producer who hails from Castledawson, Northern Ireland.
</item 7>

<item 8>
Circus Diablo
Circus Diablo is an American rock band, formed in early 2006 by Billy Morrison (vocals), Billy Duffy (lead guitar) and Ricky Warwick (rhythm guitar).

 Fuel frontman Brett Scallions and Velvet Revolver drummer Matt Sorum subsequently joined the band on bass and drums, respectively.

 To date, Circus Diablo have released one studio album, entitled "Circus Diablo".
</item 8>

<item 9>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 9>

<item 10>
The Best of Fuel
The Best of Fuel is a compilation album by American hard rock band Fuel.

 It contains songs from their first three full-length albums, "Sunburn", "Something Like Human", and "Natural Selection".

 All of the said albums featured vocalist Brett Scallions who departed from the group only a few months after the compilation's release.

 "The Best of Fuel" consists solely of the band's released singles, with the exception of "Million Miles" from "Natural Selection" (it features "Quarter" from
the same album instead).
</item 10>
=== END ITEM id=816dd144b2f11656 ===

=== ITEM id=1ca0cc4831131ac4 ===
Gold sentences and facts:
Sentence 0: In September, The Fader published an interview with CupcakKe, which she discussed her new album that would address "rape, and everything else".
  Fact 0.0: The Fader published an interview with CupcakKe in September.
  Fact 0.1: In the interview, CupcakKe discussed her new album.
  Fact 0.2: CupcakKe's new album would address "rape, and everything else".
Sentence 1: The Fader (stylized as The FADER) is a New York City-based music magazine launched in 1999 by Rob Stone and Jon Cohen, covering music, style and culture.
  Fact 1.0: The Fader, stylized as The FADER, is a New York City-based music magazine.
  Fact 1.1: The Fader was launched in 1999.
  Fact 1.2: The Fader was launched by Rob Stone and Jon Cohen.
  Fact 1.3: The Fader covers music, style and culture.
Sentence 2:  It was the first print publication to be released on iTunes.
  Fact 2.0: The Fader was the first print publication to be released on iTunes.

Retrieved items:
<item 1>
Fader Label
FADER Label is an independent record label based in New York City that has released albums by Matt and Kim, Yuna, Editors, Neon Indian, Birdmonster, and Saul
Williams.

 It was founded in 2002 by Rob Stone and Jon Cohen.
</item 1>

<item 2>
SexIs
SexIs Magazine is a quarterly print publication and daily webzine devoted to sex and sexual culture, founded in 2008 by Web Merchants, parent company of sex toy
e-tailer EdenFantasys.com.

 The first print issue debuted in November 2009, distributed nationwide as an insert in "BUST magazine".

 The website publishes articles, columns, video presentations and news items daily.
</item 2>

<item 3>
Paper (magazine)
Paper, stylized as PAPER, is a New York City-based independent magazine focusing on fashion, popular culture, nightlife, music, art and film.

 Past cover models include Kim Kardashian, Demi Lovato, Katy Perry, Miley Cyrus, Prince, CL, and Jennifer Lopez.
</item 3>

<item 4>
VIVmag
VIVmag is one of the earliest exclusively digital, interactive women's lifestyle magazines.

 Launched in 2006 and backed by Canadian industrialist David Harrison Gilmour (the founder of FIJI Water LLC), the bimonthly magazine is distributed by Zinio
Systems, Inc.

 The first issue appeared in January-February 2007.

 There is no print version.

 The paperless title can be viewed in most browsers or downloaded to your computer and read via the Zinio Reader.

 The magazine is based in Thousand Oaks, California.
</item 4>

<item 5>
State (magazine)
State.ie (formerly State Magazine) is an Irish website and formerly a monthly music magazine, which launched in April 2008 and ceased to print in January 2009
having published a total of ten issues.

 The magazine continues online and was voted Best Music Site in 2008 and Best Web Publication in 2010 in the Irish Web Awards.

 Originally the concept of the magazine involved a hard copy of which there was a charge to purchase, however after six issues it was decided to make the
magazine's content free both online and in print.

 The first issue, April 2008, appeared on Irish shelves on 6 March 2008 and featured Michael Stipe of R.E.M. on the cover.

 This immediately garnered comparisons between the new magazine's similarities with "Hot Press" who featured Stipe on their cover at the same time, a move
widely thought to be an attempt by Hot Press to stifle State's status as a serious 'alternative' to the more established local magazine.

 At a price of €5.50, "State" charged €2 more than "Hot Press".
</item 5>

<item 6>
The Fader
The Fader (stylized as The FADER) is a New York City-based music magazine launched in 1999 by Rob Stone and Jon Cohen, covering music, style and culture.

 It was the first print publication to be released on iTunes.
</item 6>

<item 7>
Exclaim!
Exclaim!

 is a monthly Canadian music magazine that features in-depth coverage of new music across all genres with a special focus on Canadian and cutting-edge artists.

 Content is based on the monthly print publication, which publishes 9 issues per year, distributing over 103,000 copies to over 2,600 locations across Canada.

 The magazine has an average of 361,200 monthly readers.

 Their website, exclaim.ca, has an average of 675,000 unique visitors a month.
</item 7>

<item 8>
Cornerstone (agency)
Cornerstone is a New York-based creative lifestyle marketing and public relations agency with offices in New York City, Los Angeles, and London.

 It is helmed by co-CEOs Rob Stone and Jon Cohen.

 The company employs approximately 100 professionals worldwide, in addition to a network of field marketing representatives throughout the United States.
</item 8>

<item 9>
Audacious (CupcakKe album)
In September, The Fader published an interview with CupcakKe, which she discussed her new album that would address "rape, and everything else".
</item 9>

<item 10>
Rap-Up
Rap-Up is a magazine launched in 2001 by founder Devin Lazerine.

 The publication was originally a website devoted to hip hop, until Lazerine decided to pitch the possibility of a magazine to several publishers.

 The magazine is focused on the hip hop and R&B aspect of the music industry, and predominantly features interviews with artists, actors and other entertainers.

 Issued quarterly, the magazine's target audience was the 14 to 28 demographic, although the demographic has become older over time.
</item 10>
=== END ITEM id=1ca0cc4831131ac4 ===

=== ITEM id=bed9c440fe1a4742 ===
Gold sentences and facts:
Sentence 0:  Born in Des Moines, Iowa, Watkins rose to fame in the early 1990s as a member of the girl-group TLC.
  Fact 0.0: Watkins was born in Des Moines, Iowa.
  Fact 0.1: Watkins rose to fame in the early 1990s.
  Fact 0.2: Watkins rose to fame as a member of the girl-group TLC.
Sentence 1:  She is best known as the former main vocalist and member of South Korean girl group T-ara.
  Fact 1.0: Watkins is best known as the former main vocalist of the South Korean girl group T-ara.
  Fact 1.1: Watkins is best known as the former member of the South Korean girl group T-ara.

Retrieved items:
<item 1>
Tionne Watkins
Tionne Tenese Watkins (born April 26, 1970), better known by her stage name T-Boz, is an American singer, songwriter, actress, author, and executive producer.

 Born in Des Moines, Iowa, Watkins rose to fame in the early 1990s as a member of the girl-group TLC.

 She has won four Grammy Awards for her work with TLC.
</item 1>

<item 2>
Kim Tae-yeon discography
Apart from soundtrack recordings, Taeyeon has also recorded duets with other artists, most notably "Like a Star" with The One and "Different" with Kim Bum-soo,
which peaked at numbers one and two on the Gaon Digital Chart, respectively.
</item 2>

<item 3>
Kim Tae-yeon discography
Kim Tae-yeon, better known by the mononym Taeyeon, is a South Korean singer.

 Her discography consists of one studio album, two extended plays (EPs), twenty singles (including four as featured artist), and three promotional singles.

 She debuted as a member of South Korean girl group Girls' Generation in August 2007 and initially gained some popularity as a singer upon recording soundtrack
songs "If" for "Hong Gil Dong" and "Can You Hear Me" for "Beethoven Virus" (2008).

 She subsequently established herself as one of the most renowned vocalist on the South Korean music scene with further soundtrack recordings, notably "I Love
You" for "" (2010), "Missing You like Crazy" for "The King 2 Hearts", "Closer" for "To the Beautiful You" (2012), and "And One" for "That Winter, the Wind
Blows" (2013), all of which managed to enter the top ten of South Korea's Gaon Digital Chart.
</item 3>

<item 4>
Kim Tae-yeon
Kim Tae-yeon (born March 9, 1989), referred to as Taeyeon, is a South Korean singer.

 She had been a trainee at S.M. Entertainment's Starlight Academy during her middle school years before debuting as a member of the agency's girl group, Girls'
Generation, in 2007.

 Since then, she has risen to prominence due to the group's success on the Asian music scene and further participated in the agency's projects Girls'
Generation-TTS and SM the Ballad.

 Aside from group activities, she has also recorded songs for various television dramas and movies.
</item 4>

<item 5>
TLC (group)
TLC is an American girl group whose original line-up consisted of Tionne "T-Boz" Watkins, Lisa "Left Eye" Lopes and Rozonda "Chilli" Thomas.

 Formed in Atlanta, Georgia in 1990, the group was very successful during the 1990s despite numerous spats with the law, each other, and the group's record
label and management.

 They scored nine top-ten hits on the "Billboard" Hot 100, including four number-one singles "Creep", "Waterfalls", "No Scrubs", and "Unpretty".

 The group also recorded four multi-platinum albums, including "CrazySexyCool" (1994) which still remains the only album by a female group to receive a diamond
certification from the Recording Industry Association of America (RIAA).

 TLC also became the first R&B group in history to receive Million certification from the Recording Industry Association of Japan (RIAJ) for "FanMail" (1999).
</item 5>

<item 6>
R U the Girl
R U the Girl (also known as R U the Girl with T-Boz and Chilli) is an American reality series that aired on UPN in 2005.

 The series featured Tionne "T-Boz" Watkins and Rozonda "Chilli" Thomas, the remaining members of the all-girl R&B group TLC whose former member, Lisa "Left
Eye" Lopes, died in a car crash in Honduras on April 25, 2002.

 Initially promoted as a contest to permanently replace Lopes 3 years after her death by TLC themselves, both Watkins and Thomas admitted that the winner of the
contest would not be joining TLC full-time and would not be a full-time replacement member; the winner would only provide guest vocals on a new single by the
duo.
</item 6>

<item 7>
Park So-yeon (singer)
Park So-yeon (born October 5, 1987), better known mononymously as Soyeon, is a South Korean singer and actress.

 She is best known as the former main vocalist and member of South Korean girl group T-ara.
</item 7>

<item 8>
Kim Tae-yeon (actress)
Kim Tae-yeon (born January 3, 1976) is a South Korean actress.

 She began her entertainment career as a model, winning Model Line's 40th Fashion Model contest in 1996 and the Pantene Model contest sponsored by Ford Models
in 2000.

 Kim made her film debut in the highly controversial film "Lies" in 1999.
</item 8>

<item 9>
Rain (Taeyeon song)
"Rain" is a song by South Korean singer Kim Tae-yeon, a member of the South Korean girl group Girls' Generation.

 It was released digitally by S.M. Entertainment on February 3, 2016.
</item 9>

<item 10>
As One (musical duo)
As One () is a South Korean R&B duo consisting of Korean-American singers Lee Min-young (known as Lee Min) and Chae Da-hee (known as Crystal).

 They are signed to record label Brand New Music.
</item 10>
=== END ITEM id=bed9c440fe1a4742 ===

=== ITEM id=223c77031d63e966 ===
Gold sentences and facts:
Sentence 0: Robert Paul Holdstock (2 August 1948 – 29 November 2009) was an English novelist and author best known for his works of Celtic, Nordic, Gothic and Pictish fantasy literature, predominantly in the fantasy subgenre of mythic fiction.
  Fact 0.0: Robert Paul Holdstock was born on 2 August 1948.
  Fact 0.1: Robert Paul Holdstock died on 29 November 2009.
  Fact 0.2: Robert Paul Holdstock was an English novelist.
  Fact 0.3: Robert Paul Holdstock was an English author.
  Fact 0.4: Robert Paul Holdstock was best known for his works of Celtic fantasy literature.
  Fact 0.5: Robert Paul Holdstock was best known for his works of Nordic fantasy literature.
  Fact 0.6: Robert Paul Holdstock was best known for his works of Gothic fantasy literature.
  Fact 0.7: Robert Paul Holdstock was best known for his works of Pictish fantasy literature.
  Fact 0.8: Robert Paul Holdstock's works are predominantly in the fantasy subgenre of mythic fiction.
Sentence 1: Henry Louis Mencken (September 12, 1880 – January 29, 1956) was an American journalist, satirist, cultural critic and scholar of American English.
  Fact 1.0: Henry Louis Mencken was born on September 12, 1880.
  Fact 1.1: Henry Louis Mencken died on January 29, 1956.
  Fact 1.2: Henry Louis Mencken was an American journalist.
  Fact 1.3: Henry Louis Mencken was an American satirist.
  Fact 1.4: Henry Louis Mencken was an American cultural critic.
  Fact 1.5: Henry Louis Mencken was a scholar of American English.

Retrieved items:
<item 1>
The American Mercury
The American Mercury was an American magazine published from 1924 to 1981.

 It was founded as the brainchild of H. L. Mencken and drama critic George Jean Nathan.

 The magazine featured writing by some of the most important writers in the United States through the 1920s and 1930s.

 After a change in ownership in the 1940s, the magazine attracted conservative writers.

 A second change in ownership a decade later turned the magazine into a virulently anti-Semitic publication.

 It was published monthly in New York City.

 The magazine went out of business in 1981, having spent the last 25 years of its existence in decline and controversy.
</item 1>

<item 2>
Oswalt Kolle
Oswalt Kolle (2 October 1928, Kiel – 24 September 2010, Amsterdam) was a German sex educator, who became famous during the late 1960s and early 1970s for his
numerous pioneering books and films on human sexuality.

 His work was translated into all major languages, while his films found an audience of 140 million worldwide.

 In his 1997 book "Open to Both Sides" he came out as bisexual.

 He was awarded the Magnus Hirschfeld Medal in 2000.
</item 2>

<item 3>
Christopher Guest
Christopher Haden-Guest, 5th Baron Haden-Guest (born February 5, 1948), simply known as Christopher Guest, is an English-American screenwriter, composer,
musician, director, actor, and comedian who holds dual British and American citizenship.

 Guest is most widely known in Hollywood for having written, directed and starred in his series of comedy films shot in mock-documentary (mockumentary) style.

 Many scenes and character backgrounds in Guest's films are written and directed, although actors have no rehearsal time and the ensemble improvise scenes while
filming them.

 The series of films began with "This Is Spinal Tap" (directed by Rob Reiner), and continued with "Waiting for Guffman", "Best In Show", "A Mighty Wind", "For
Your Consideration", and "Mascots".
</item 3>

<item 4>
John Updike
John Hoyer Updike (March 18, 1932 – January 27, 2009) was an American novelist, poet, short story writer, art critic, and literary critic.

 One of only three writers to win the Pulitzer Prize for Fiction more than once (the others were Booth Tarkington and William Faulkner), Updike published more
than twenty novels, more than a dozen short-story collections, as well as poetry, art and literary criticism and children's books during his career.
</item 4>

<item 5>
Robert Holdstock
Robert Paul Holdstock (2 August 1948 – 29 November 2009) was an English novelist and author best known for his works of Celtic, Nordic, Gothic and Pictish
fantasy literature, predominantly in the fantasy subgenre of mythic fiction.
</item 5>

<item 6>
Lionel Snell
Ramsey Dukes is the current and most well-known pen name of Lionel Snell, a contemporary English magician, publisher and author on magic and philosophy.

 He has also under the pen names Lemuel Johnston.

 Angerford and Lea, Adamai Philotunus and Per Anum Ad Astra.

 His regular satirical column "The Satanist's Diary" was published in the magazine "Aquarian Arrow" during the 1980s and early 1990s under the pen name The Hon
Hugo C StJ l'Estrange and included a multitude of other pen names as contributors.
</item 6>

<item 7>
H. L. Mencken
Henry Louis Mencken (September 12, 1880 – January 29, 1956) was an American journalist, satirist, cultural critic and scholar of American English.

 Known as the "Sage of Baltimore", he is regarded as one of the most influential American writers and prose stylists of the first half of the twentieth century.

 He commented widely on the social scene, literature, music, prominent politicians and contemporary movements.

 His satirical reporting on the Scopes trial, which he dubbed the "Monkey Trial", also gained him attention.
</item 7>

<item 8>
Robert Holdstock bibliography
This is a bibliography of fantasy author Robert Holdstock.
</item 8>

<item 9>
Adam Fox
Adam Fox (1883 – 1977), Canon, was the Dean of Divinity at Magdalen College, Oxford.

 He was one of the first members of the "Inklings", a literary group which also included C.S. Lewis and J.R.R. Tolkien.

 Between 1938 and 1942 he was Professor of Poetry.

 Later he became Canon of Westminster Abbey and he is buried there in Poet's Corner.

 He was also Warden (Headmaster) of the famous Radley College.
</item 9>

<item 10>
Stanley Bing
Stanley Bing is the pen name of Gil Schwartz (born May 20, 1951 in New York, NY), a business humorist and novelist.

 He has written a column for "Fortune" magazine for more than twenty years, after having spent a decade at "Esquire".

 He is the author of thirteen books including "What Would Machiavelli Do?"

 and "The Curriculum", a satirical textbook for a business school that also offers lessons on the Web.

 Schwartz is the senior executive vice president of corporate communications and Chief Communications Officer for CBS.
</item 10>
=== END ITEM id=223c77031d63e966 ===

=== ITEM id=d226f0cc6e8faf09 ===
Gold sentences and facts:
Sentence 0: Tender Fictions is a 1996 autobiographical documentary film directed by American experimental filmmaker Barbara Hammer.
  Fact 0.0: Tender Fictions is a 1996 autobiographical documentary film.
  Fact 0.1: Tender Fictions was directed by American experimental filmmaker Barbara Hammer.
Sentence 1: Bougainville – Our Island Our Fight is a 1998 Australian documentary film.
  Fact 1.0: Bougainville – Our Island Our Fight is a 1998 Australian documentary film.
Sentence 2:  It was produced and directed by Wayne Coles-Janess.
  Fact 2.0: Bougainville – Our Island Our Fight was produced by Wayne Coles-Janess.
  Fact 2.1: Bougainville – Our Island Our Fight was directed by Wayne Coles-Janess.

Retrieved items:
<item 1>
Flight of the Red Balloon
Flight of the Red Balloon (French: Le voyage du ballon rouge ) is a 2007 French-Taiwanese film directed by Hou Hsiao-hsien.

 It is the first part in a new series of films produced by Musée d'Orsay, and tells the story of a French family as seen through the eyes of a Chinese student.

 The film was shot in August and September 2006 on location in Paris.

 This is Hou Hsiao-hsien's first non-Asian film.

 It references the classic 1956 French short "The Red Balloon" directed by Albert Lamorisse.
</item 1>

<item 2>
Wayne Coles-Janess
Wayne Coles-Janess is an Australian producer, writer and director of drama and documentary film and TV programs.

 Based in Melbourne, Australia, he has produced documentaries about frontier places in the country.

 He has also made some documentaries in several international locations, including during times of war.
</item 2>

<item 3>
From the Sea to the Land Beyond
From The Sea to the Land Beyond: Britain's Coast on Film is a documentary feature film directed by Penny Woolcock, with an original soundtrack by British
indie-rock band British Sea Power.

 The project was originally produced by Crossover and Sheffield Doc/Fest as part of The Space project from the BBC and the Arts Council England.

 The film was edited by Alex Fry.
</item 3>

<item 4>
Bougainville – Our Island Our Fight
Bougainville – Our Island Our Fight is a 1998 Australian documentary film.

 It was produced and directed by Wayne Coles-Janess.
</item 4>

<item 5>
Waiting for Happiness
The film presents typical Mauritanian moments of beauty, struggle, alienation, and humor, which are experienced by groups socially divided from each other, such
as Bidhan women drinking tea and gossiping, West African migrants passing through Mauritania to get to Europe (and finding an unsuccessful comrade washed
ashore).

 The young protagonist who has returned interacts with all of these groups as an outsider, as he struggles to remember even his own Hassaniya Arabic dialect,
but prefers instead French.

 Many of the themes and characters presage Sissako's 2014 film Timbuktu, and both explore liminal Sahel identities authentically situated in everyday life.

 Waiting for Happiness premiered at the 2002 Cannes Film Festival in the Un Certain Regard section.
</item 5>

<item 6>
In the Shadow of the Palms
In the Shadow of the Palms (2005) is a documentary produced and directed by the Australian filmmaker Wayne Coles-Janess.

 He filmed it in Iraq prior to, during and after the fall of Saddam Hussein after the United States invasion of 2003.

 An Arab-language film, it documents the changes in Iraqi society and the lives of ordinary Iraqis by focusing on a cross-section of individuals.
</item 6>

<item 7>
The Boys in Company C
The Boys in Company C, directed by Sidney J. Furie, starring Stan Shaw, Andrew Stevens, Craig Wasson, James Canning, and Michael Lembeck, is a 1978 film about
United States Marine Corps recruits preparing for duty, and their subsequent combat in the Vietnam War.

 It was among the first Vietnam War films to appear after the Vietnam Era, and was also the first role for R. Lee Ermey of "Full Metal Jacket" fame.

 "The Boys in Company C" is the first in Furie's Vietnam War motion picture trilogy, followed by 2001's "Under Heavy Fire" and 2006's "The Veteran".
</item 7>

<item 8>
Tender Fictions
Tender Fictions is a 1996 autobiographical documentary film directed by American experimental filmmaker Barbara Hammer.

 It is the second of a trilogy of documentary films that includes "Nitrate Kisses" and "History Lessons".

 Together, the three films are sometimes known as the "History trilogy".

 "Tender Fictions" details Hammer's life and her attempts to "construct" a self.

 The film was nominated for a prize at the 1996 Sundance Film Festival.
</item 8>

<item 9>
The Coconut Revolution
The Coconut Revolution is a 2001 multi-award winning documentary film about the struggle of the indigenous peoples of Bougainville Island during the
Bougainville Civil War.

 The movement is described as the "world's first successful eco-revolution" and has drawn parallels with the conflict depicted in the 2009 film, "Avatar".
</item 9>

<item 10>
Beyond This Place (2010 film)
Beyond This Place (2010) is a documentary film directed by Kaleo La Belle, screened at a number of film festivals.

 In it, La Belle reunites with his biological father Cloud Rock La Belle, a charismatic figure who was largely absent from his life.

 The film deals, in a personal and intimate way, with issues of parenting, issues of freedom versus responsibility, and with the aging of the 60's generation.
</item 10>
=== END ITEM id=d226f0cc6e8faf09 ===

=== ITEM id=be14b0bf56abaf3e ===
Gold sentences and facts:
Sentence 0:  It began in 1995 as the merger of two take-and-bake pizza companies: Papa Aldo's Pizza (founded in 1981) and Murphy's Pizza (founded in 1981).
  Fact 0.0: The company began in 1995.
  Fact 0.1: The company began as the merger of two take-and-bake pizza companies.
  Fact 0.2: One of the merged companies was Papa Aldo's Pizza.
  Fact 0.3: Papa Aldo's Pizza was founded in 1981.
  Fact 0.4: The other merged company was Murphy's Pizza.
  Fact 0.5: Murphy's Pizza was founded in 1981.
Sentence 1: Gino's Pizza and Spaghetti is a restaurant chain with 40 locations, most of them within the U.S. state of West Virginia.
  Fact 1.0: Gino's Pizza and Spaghetti is a restaurant chain.
  Fact 1.1: Gino's Pizza and Spaghetti has 40 locations.
  Fact 1.2: Most of Gino's Pizza and Spaghetti's locations are within the U.S. state of West Virginia.
Sentence 2:  The company was founded by Kenney Grant in 1961.
  Fact 2.0: The company was founded by Kenney Grant.
  Fact 2.1: The company was founded in 1961.

Retrieved items:
<item 1>
Panera Bread is a U.S.–Canada bakery-café chain headquartered near St. Louis, where it is called Saint Louis Bread Company. Indian entrepreneur Prem Ganapathy
founded Dosa Plaza, growing it to 45 outlets across four countries. Quebec’s independently franchised La Belle Province is sometimes open 24 hours. Gino’s
Hamburgers began in Baltimore in 1957 and relaunched in 2010. West Virginia-based Tudor’s Biscuit World serves breakfast and homestyle
</item 1>

<item 2>
Gino's Pizza and Spaghetti
Gino's Pizza and Spaghetti is a restaurant chain with 40 locations, most of them within the U.S. state of West Virginia.

 The company was founded by Kenney Grant in 1961.

 Many locations are shared with Tudor's Biscuit World although the Gino's brand is exclusive to West Virginia.

 There is one located in Ohio, while there are stand alone Tudor's locations in eastern Kentucky, southern Ohio and southwest Virginia.

 Gino's serves pizza, spaghetti, sandwiches, and more.

 Company headquarters are located in Huntington, West Virginia and Nitro, West Virginia.
</item 2>

<item 3>
Gino's Hamburgers
Gino's Hamburgers was a fast-food restaurant chain founded in Baltimore, Maryland, by Baltimore Colts defensive end Gino Marchetti and running back Alan Ameche,
along with their close friend Louis Fischer, in 1957.

 A new group of restaurants under the Gino's name involving some of the principals of the original chain was started in 2010.
</item 3>

<item 4>
Gino's East
Gino's East is a Chicago-based restaurant chain, notable for its deep-dish pizza (sometimes called Chicago-style pizza), and for its interior walls, which
patrons have covered in graffiti and etchings.

 The restaurant features deep-dish pizza baked in cast-iron pans, as well as sandwiches, soups and salads.
</item 4>

<item 5>
Papa Gino's
Papa Gino's, Inc. is a restaurant chain based in Dedham, Massachusetts specializing in American-style pizza along with pasta, subs, salads, and a variety of
appetizers.

 There are over 150 Papa Gino's locations in Connecticut, Maine, Massachusetts, New Hampshire, and Rhode Island.
</item 5>

<item 6>
Papa Murphy's
Papa Murphy's, a business based in Vancouver, Washington, United States, is a take-and-bake pizza company.

 It began in 1995 as the merger of two take-and-bake pizza companies: Papa Aldo's Pizza (founded in 1981) and Murphy's Pizza (founded in 1981).

 The company and its franchisees operate more than 1,300 outlets in the United States and Canada.

 Papa Murphy's is the fifth-largest pizza chain in the United States.
</item 6>

<item 7>
John Murphy (fiddle player)
John (Boss) Murphy (1875–1955) from The Leap, Churchtown, Co.

 Cork was a former farmer by profession, but possessed a keen interest in fiddle playing, and was renowned locally for his ability on the instrument.

 His father, William Murphy (1829–1911), was both a fiddle player and maker, indeed John was the proud owner of an instrument that had been made by his father.

 John’s father was his first fiddle teacher.

 His siblings, a brother and three sisters, also played the instrument but their interest in music waned as they reached adulthood and they did not continue to
play.

 John’s musical literacy was also gained primarily from his father, who had learnt to read music at a hedge school at Ballygrace in the locality taught by a
Thomas Croke.

 The same Thomas Croke, many years later, stayed for long periods of time in the Murphy household and would undoubtedly also have taught the young John
directly.
</item 7>

<item 8>
Pizza 73
Pizza 73 is a Canadian restaurant chain that offers a number of different styles of pizza, along with chicken wings.

 It has been operated by Pizza Pizza since 2007.

 Toronto-based Pizza Pizza had acquired the restaurant for a total of $CAN70.2 million.

 There are 89 locations throughout Western Canada, which include the provinces of British Columbia, Alberta, and Saskatchewan.

 The restaurant's name originates from its original phone number: 473–7373.

 Founded by David Tougas and Guy Goodwin in 1985, Pizza 73 is headquartered in Edmonton, Alberta, Canada.
</item 8>

<item 9>
Lucky Luciano
Charles "Lucky" Luciano ( ; born Salvatore Lucania November 24, 1897 – January 26, 1962) was an Italian-American mobster and crime boss.

 Luciano is considered the father of modern organized crime in the United States for the establishment of the first Commission.

 He was also the first official boss of the modern Genovese crime family.

 He was, along with his associates, instrumental in the development of the National Crime Syndicate.
</item 9>

<item 10>
Prem Ganapathy
Prem Ganapathy is an Indian entrepreneur and businessman.

 He is the founder of the restaurant chain Dosa plaza.

 Starting with meager investment, he expanded Dosa plaza into a restaurant chain with 45 outlets in India, New Zealand, Oman, and UAE.
</item 10>
=== END ITEM id=be14b0bf56abaf3e ===

=== ITEM id=5efa453a29cab620 ===
Gold sentences and facts:
Sentence 0:  It has its offices in New Roc City in New Rochelle, New York.
  Fact 0.0: An entity has its offices in New Roc City in New Rochelle, New York.
Sentence 1: New Roc City, also known as New Rochelle Center, is an entertainment, retail and residential complex in the Downtown section of the city of New Rochelle in Westchester County, New York.
  Fact 1.0: New Roc City is also known as New Rochelle Center.
  Fact 1.1: New Roc City is an entertainment complex.
  Fact 1.2: New Roc City is a retail complex.
  Fact 1.3: New Roc City is a residential complex.
  Fact 1.4: New Roc City is in the Downtown section of the city of New Rochelle.
  Fact 1.5: New Rochelle is in Westchester County, New York.

Retrieved items:
<item 1>
Ware's Department Store
Ware’s Department Store is a historic building located in the city of New Rochelle in Westchester County, New York and is significant for both architectural and
commercial reasons.

 Ware's was Westchester’s first and, for many years, largest department store, and was prominently located on New Rochelle’s fashionable Main Street.

 The store operated from 1881 to the late 1930s, when the property was sold to the retailer Bloomingdale's to serve as their first suburban department store
location.

 Not only was Ware's a key place in New Rochelle during its 20th-century boom years, its founder and owner, Howard R. Ware was a leading figure in the rapidly
growing community as well.

 Ware first moved to New Rochelle from Massachusetts at the age of 13 and began to work as a clerk.

 In 1881 he became partner in the firm of Ware & Sheffield, which eventually became a stock company in 1913.

 renamed He was a director and vice president of the National City Bank of New Rochelle, a founder and first president of the local Y.M.C.A. from 1899 to 1916,
and an active member of St.
</item 1>

<item 2>
Japanese Weekend School of New York
The Japanese Weekend School of New York (JWSNY; ニューヨーク "Nyūyōku Hoshū Jugyō Kō") is a Japanese supplementary school in the New York City metropolitan area.

 It has its offices in New Roc City in New Rochelle, New York.

 The Japanese Educational Institute of New York (JEI; ニューヨーク日本人教育審議会 "Nyūyōku Nihonjin Kyōiku Shingi Kai") manages the school system, and the JWSNY is one of
its two weekend school systems.

 The JEI also operates two Japanese day schools in the New York area.
</item 2>

<item 3>
Montefiore New Rochelle Hospital
Montefiore New Rochelle Hospital (formerly Sound Shore Medical Center) is a community-based, teaching hospital located at 16 Guion Place in the city of New
Rochelle, in Westchester County, New York, and affiliated with the Albert Einstein College of Medicine.

 The hospital opened in 1892 as New Rochelle Hospital.

 On November 6, 2013, Sound Shore was acquired by the Bronx-based Montefiore Medical Center, the University Hospital for the Albert Einstein College of
Medicine, was renamed Montefiore New Rochelle Hospital, and became part of the Montefiore Health System.
</item 3>

<item 4>
Princeton Community Japanese Language School
The Princeton Community Japanese Language School (PCJLS; プリンストン日本語学校 "Purinsuton Nihongo Gakkō") is a Japanese weekend school in the Princeton, New Jersey area.

 It holds weekend Japanese classes for Japanese citizen children abroad to the standard of the Ministry of Education, Culture, Sports, Science and Technology
(MEXT), and it also has classes for people with Japanese as a second language.
</item 4>

<item 5>
New Roc City
New Roc City, also known as New Rochelle Center, is an entertainment, retail and residential complex in the Downtown section of the city of New Rochelle in
Westchester County, New York.

 It is located at 33 LeCount Place, between Main Street North and Main Street South.

 The center was built on the site of the former New Rochelle Mall which closed in 1992.
</item 5>

<item 6>
Standard Star Building
The Standard Star Building is a historic commercial structure located in the Downtown section of New Rochelle, Westchester County, New York.

 The building, designed by architect Lawrence J. Barnard, was completed in 1924 and is an architecturally significant example of the Italian Renaissance style
in New Rochelle.

 Although it has been altered, these changes occurred only on the Le Count Place façade and the interior of the building.

 The original Standard Star building exterior remains largely unchanged.

 It is further historically significant for its association with a long-published New Rochelle newspaper covering life in New Rochelle.
</item 6>

<item 7>
San Francisco Japanese School
The San Francisco Japanese School (SFJS, サンフランシスコ日本語 "San Furanshisuko Nihongo Hoshū Kō") is a weekend Japanese school serving the San Francisco Bay Area.

 The system, with its administrative offices in San Francisco, is a 501(c)(3) non-profit organization, and was the world's second largest overseas Japanese
weekend school in 2006.

 The school is supported by the Japanese government.

 It is known as the best Japanese supplementary school in the world with mostly great teachers.

 Although it does have a shortage of teachers due to the large number of students.
</item 7>

<item 8>
New Jersey Japanese School
The New Jersey Japanese School (ニュージャージー日本人学校 , "Nyūjājī Nihonjin Gakkō" , NJJS) is a Japanese school located in Oakland, New Jersey, United States in the New
York City metropolitan area.

 It is one of the two Japanese day schools operated by the Japanese Educational Institute of New York (JEI; ニューヨーク日本人教育審議会 "Nyūyōku Nihonjin Kyōiku Shingi
Kai"), a nonprofit organization which also operates two Japanese weekend school systems in the New York City area.
</item 8>

<item 9>
New Rochelle Walk of Fame
The New Rochelle Walk of Fame was installed in 2011 in Ruby Dee Park at Library Green, located in the downtown section of the city of New Rochelle in
Westchester County, New York.

 The "walk" is a tribute to some of New Rochelle's most notable residents from throughout its 325-year history.

 It was created and funded by former resident Roderick Kennedy, Jr., working in partnership with the City of New Rochelle and the New Rochelle Business
Improvement District.
</item 9>

<item 10>
New Rochelle Harbor (Long Island Sound)
New Rochelle Harbor is the name of a harbor located along Long Island Sound in the city of New Rochelle in Westchester County, New York.

 The Davenport Neck peninsula off the mainland divides New Rochelle's waterfront into two bays; the westerly referred to as New Rochelle Harbor and the easterly
as Echo Bay.
</item 10>
=== END ITEM id=5efa453a29cab620 ===

=== ITEM id=6551e000e3f436de ===
Gold sentences and facts:
Sentence 0:  One of only three writers to win the Pulitzer Prize for Fiction more than once (the others were Booth Tarkington and William Faulkner), Updike published more than twenty novels, more than a dozen short-story collections, as well as poetry, art and literary criticism and children's books during his career.
  Fact 0.0: Updike was one of only three writers to win the Pulitzer Prize for Fiction more than once.
  Fact 0.1: The other two writers who won the Pulitzer Prize for Fiction more than once were Booth Tarkington and William Faulkner.
  Fact 0.2: Updike published more than twenty novels during his career.
  Fact 0.3: Updike published more than a dozen short-story collections during his career.
  Fact 0.4: Updike published poetry during his career.
  Fact 0.5: Updike published art and literary criticism during his career.
  Fact 0.6: Updike published children's books during his career.
Sentence 1:  Seventeen of his novels were bestsellers, and more than 100 million copies of his books are in print.
  Fact 1.0: Seventeen of Updike's novels were bestsellers.
  Fact 1.1: More than 100 million copies of Updike's books are in print.

Retrieved items:
<item 1>
Stephen R. Donaldson
Stephen Reeder Donaldson (born May 13, 1947) is an American fantasy, science fiction and mystery novelist, most famous for "The Chronicles of Thomas Covenant",
his ten-novel fantasy series.

 His work is characterized by psychological complexity, conceptual abstractness, moral bleakness, and the use of an arcane vocabulary, and has attracted
critical praise for its "imagination, vivid characterizations, and fast pace".

 He earned his bachelor's degree from The College of Wooster and a Master's degree from Kent State University.

 He currently resides in New Mexico.
</item 1>

<item 2>
John Updike
John Hoyer Updike (March 18, 1932 – January 27, 2009) was an American novelist, poet, short story writer, art critic, and literary critic.

 One of only three writers to win the Pulitzer Prize for Fiction more than once (the others were Booth Tarkington and William Faulkner), Updike published more
than twenty novels, more than a dozen short-story collections, as well as poetry, art and literary criticism and children's books during his career.
</item 2>

<item 3>
David Ignatius
David R. Ignatius (May 26, 1950), is an American journalist and novelist.

 He is an associate editor and columnist for "The Washington Post".

 He also co-hosts PostGlobal, an online discussion of international issues at Washingtonpost.com, with Fareed Zakaria.

 He has written nine novels, including "Body of Lies", which director Ridley Scott adapted into a film.

 He is a former Adjunct Lecturer at the Kennedy School of Government at Harvard University and currently Senior Fellow to the Future of Diplomacy Program.

 He has received numerous honors, including the Legion of Honor from the French Republic, the Urbino World Press Award from the Italian Republic, and a lifetime
achievement award from the International Committee for Foreign Journalism.
</item 3>

<item 4>
Tom Clancy's Net Force Explorers: Gameprey
Tom Clancy's Net Force Explorers or Net Force Explorers is a series of young adult novels created by Tom Clancy and Steve Pieczenik as a spin-off of the
military fiction series Tom Clancy's Net Force.
</item 4>

<item 5>
Tom Clancy's Net Force Explorers: High Wire
Tom Clancy's Net Force Explorers or Net Force Explorers is a series of young adult novels created by Tom Clancy and Steve Pieczenik as a spin-off of the
military fiction series Tom Clancy's Net Force.
</item 5>

<item 6>
Tom Clancy's Rainbow 6: Patriots
Tom Clancy's Rainbow 6: Patriots is a cancelled first-person shooter video game, part of the "Tom Clancy's Rainbow Six" series, announced on the cover of the
December 2011 issue of "Game Informer".

 It was to be published by Ubisoft, and was developed by the company's Montreal studio, with additional development by Ubisoft Toronto and Red Storm
Entertainment.

 Due to the death of Tom Clancy in October 2013, concern was raised that this game would become the last to bear his name.

 Ubisoft has since stated that they will continue putting Tom Clancy's name on future Tom Clancy titles out of respect for the late author.
</item 6>

<item 7>
Tom Clancy's Net Force Explorers: Deathworld
Tom Clancy's Net Force Explorers or Net Force Explorers is a series of young adult novels created by Tom Clancy and Steve Pieczenik as a spin-off of the
military fiction series Tom Clancy's Net Force.
</item 7>

<item 8>
Tom Clancy's Net Force Explorers: Death Match
Tom Clancy's Net Force Explorers or Net Force Explorers is a series of young adult novels created by Tom Clancy and Steve Pieczenik as a spin-off of the
military fiction series Tom Clancy's Net Force.
</item 8>

<item 9>
Tom Clancy's Net Force Explorers: Duel Identity
Tom Clancy's Net Force Explorers or Net Force Explorers is a series of young adult novels created by Tom Clancy and Steve Pieczenik as a spin-off of the
military fiction series Tom Clancy's Net Force.
</item 9>

<item 10>
Tom Clancy
Thomas Leo "Tom" Clancy Jr. (April 12, 1947 – October 1, 2013) was an American novelist best known for his technically detailed espionage and military-science
story lines set during and after the Cold War.

 Seventeen of his novels were bestsellers, and more than 100 million copies of his books are in print.

 His name was also used on movie scripts written by ghost writers, nonfiction books on military subjects, and video games.

 He was a part-owner of the Baltimore Orioles and vice-chairman of their community activities and public affairs committees.
</item 10>
=== END ITEM id=6551e000e3f436de ===
