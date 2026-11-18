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

This file has 13 items.

=== ITEM id=f8b58b04d7735dd5 ===
Gold sentences and facts:
Sentence 0: Eryngium is a genus of flowering plants in the family Apiaceae.
  Fact 0.0: Eryngium is a genus of flowering plants.
  Fact 0.1: Eryngium is in the family Apiaceae.
Sentence 1: Arnebia is a genus of flowering plants in the family Boraginaceae.
  Fact 1.0: Arnebia is a genus of flowering plants.
  Fact 1.1: Arnebia is in the family Boraginaceae.

Retrieved items:
<item 1>
Amarylloidinae
Amarylloidinae is a now obsolete informal name for an "infrafamily" within the Amaryllidaceae (Amaryllis) family, erected by Hamilton Traub.

 This grouping was designed to fill a perceived gap between the formal rank of subfamily and tribes.

 In his treatment of this family, he divided it first into four subfamiles.

 Within subfamily Amarylloideae he then divided his sixteen tribes into two infrafamilies, Amarylloidinae (12 tribes) and Pancratioidinae (4 tribes), both of
which were subsequently demonstrated to be polyphyletic, and hence were abandoned by Dahlgren, who used no rank between family and tribe.

 On the other hand, he also used a much more restricted Amaryllidaceae corresponding to Traub's subfamily Amarylloideae.

 Thus Traub's Amarylloideae most closely resembles subfamily Amaryllidoideae "sensu" APGIII.
</item 1>

<item 2>
Eucryphia
Eucryphia is a small genus of trees and large shrubs native to the south temperate regions of South America and coastal eastern Australia.

 Sometimes placed in a family of their own, the Eucryphiaceae, more recent classifications place them in the Cunoniaceae.

 There are seven species, two in South America and five in Australia, and several named hybrids.

 They are mostly evergreen though one species ("E. glutinosa") is usually deciduous.
</item 2>

<item 3>
Ericaceae
The Ericaceae are a family of flowering plants, commonly known as the heath or heather family, found most commonly in acid and infertile growing conditions.

 The family is large, with c. 4250 known species spread across 124 genera, making it the 14th most species-rich family of flowering plants.

 The many well-known and economically important members of the Ericaceae include the cranberry, blueberry, huckleberry, rhododendron (including azaleas), and
various common heaths and heathers ("Erica", "Cassiope", "Daboecia", and "Calluna" for example).
</item 3>

<item 4>
Eriogonum
Eriogonum is the scientific name for a genus of flowering plants in the family Polygonaceae.

 The genus is found in North America and is known as wild buckwheat.

 This is a highly species-rich genus, and indications are that active speciation is continuing.

 It includes some common wildflowers such as the California buckwheat ("Eriogonum fasciculatum").
</item 4>

<item 5>
Catuaba
The name catuaba (pronounced [ka.two.

'aba], a Guarani word that means "what gives strength to the Indian") is used for the infusions of the bark of a number of trees native to Brazil.

 The most widely used barks are derived from the trees "Trichilia catigua" and "Erythroxylum vaccinifolium".

 Other catuaba preparations use the bark of trees from the following genera or families: "Anemopaegma", "Ilex", "Micropholis", "Phyllanthus", "Secondatia",
"Tetragastris" and species from the Myrtaceae.
</item 5>

<item 6>
Ardisia
Ardisia (coralberry or marlberry) is a genus of flowering plants in the family Primulaceae.

 It was in the former Myrsinaceae family now recognised as the myrsine sub-family Myrsinoideae.

 They are distributed in the Americas, Asia, Australia, and the Pacific Islands, mainly in the tropics.

 There are 400 to 500 species.
</item 6>

<item 7>
Eryngium
Eryngium is a genus of flowering plants in the family Apiaceae.

 There are about 250 species.

 The genus has a cosmopolitan distribution, with the center of diversity in South America.

 Common names include eryngo and amethyst sea holly (though the genus is not related to the true hollies, "Ilex").
</item 7>

<item 8>
Cyanobiont
Several cyanobionts involved with fungi and marine organisms also belong to the genera "Richelia, Calothrix, Synechocystis, Aphanocapsa" and "Anabaena", as well
as the species "Oscillatoria spongeliae".

 Although there are many documented symbioses between cyanobacteria and marine organisms, little is known about the nature of many of these symbioses.

 The possibility of discovering more novel symbiotic relationships is apparent from preliminary microscopic observations.
</item 8>

<item 9>
Lophodermium
Lophodermium is a genus of fungi within the family Rhytismataceae.

 The genus contains 145 species and has a global distribution.

 Species of this genus are usually observed producing zone lines, conidiomata and ascomata on dead fallen leaves, but at least some are known to colonize living
leaves.

 In many cases they then live inside the colonized leaf as a symptomless endobiont, where they are regarded as detritivores utilising dead plant matter.

 In a few cases they may kill all or part of the leaf prematurely, and there is a substantial literature dealing with those species as plant pathogens.

 The genus infects many different plant families but with a notable concentration in the family Pinaceae; many "Lophodermium" species are restricted to a single
host genus (or even species), but some, particularly those infecting grasses, may infect several genera.

 Some are economically important plant pathogens, such as those that cause needlecast disease in European Black Pine, Scots Pine and Red Pine in forestry and
christmas tree plantations.
</item 9>

<item 10>
Arnebia
Arnebia is a genus of flowering plants in the family Boraginaceae.
</item 10>
=== END ITEM id=f8b58b04d7735dd5 ===

=== ITEM id=30e8b393c1c2a448 ===
Gold sentences and facts:
Sentence 0: The City Heiress is a play by Aphra Behn produced in 1682.
  Fact 0.0: The City Heiress is a play by Aphra Behn.
  Fact 0.1: The City Heiress was produced in 1682.
Sentence 1: Aphra Behn ( ; 14 December 1640?
  Fact 1.0: Aphra Behn's birth date is listed as 14 December 1640, marked with a question mark as uncertain.
Sentence 2:  (baptismal date)–16 April 1689) was a British playwright, poet, translator and fiction writer from the Restoration era.
  Fact 2.0: The 14 December 1640 date for Aphra Behn is a baptismal date.
  Fact 2.1: Aphra Behn died on 16 April 1689.
  Fact 2.2: Aphra Behn was a British playwright.
  Fact 2.3: Aphra Behn was a poet.
  Fact 2.4: Aphra Behn was a translator.
  Fact 2.5: Aphra Behn was a fiction writer.
  Fact 2.6: Aphra Behn was from the Restoration era.

Retrieved items:
<item 1>
Betty Ann Davies
Betty Ann Davies (24 December 1910 – 14 May 1955) was a British stage and film actress active from the 1920s to the 1950s.

 She died young in 1955 aged 44 following an operation.

 Miss Davies made her first stage appearance at the Palladium in a revue in 1924.

 The following year she joined Cochran's Young Ladies in revues such as "One Dam Thing After Another" and "This Year of Grace".

 Betty Ann Davies enjoyed a long and distinguished West End career which included "The Good Companions" 1934, "Morning Star" 1942, and "Blithe Spirit" 1943.

 Her outstanding stage triumph was in the role of Blanche du Bois, which she took over from Vivien Leigh, in the original West End production of " A Streetcar
Named Desire".

 Miss Davis appeared in 38 films, and was active in TV at the time of her death.

 She left one son Brook Blackford.
</item 1>

<item 2>
Marisha Chamberlin
Marisha Anne Chamberlain (born January 6, 1952) is an American writer.

 Her most recent work, the libretto for "Mortals & Angels", a collaboration with American composer Carol Barnett, premiered at Carnegie Hall in 2016, a
companion piece to their widely produced collaborative piece, "The World Beloved: A Bluegrass Mass".

 Her stage plays, both original works and adaptations are widely produced.

 Her debut novel, "The Rose Variations", was published by Soho Press in 2009.

 Her play, "Scheherazade", won the Dramatists Guild/CBS National Award, and in her screenplay version, played on public television across the country, and was
screened at the British Film Institute Festival in the category of Best of American Public Television.
</item 2>

<item 3>
Aphra Behn
Aphra Behn ( ; 14 December 1640?

 (baptismal date)–16 April 1689) was a British playwright, poet, translator and fiction writer from the Restoration era.

 As one of the first English women to earn her living by her writing, she broke cultural barriers and served as a literary role model for later generations of
women authors.

 Rising from obscurity, she came to the notice of Charles II, who employed her as a spy in Antwerp.

 Upon her return to London and a probable
</item 3>

<item 4>
Elswyth Thane
Helen Elswyth Thane Ricker Beebe (May 16, 1900 – July 31, 1984) was an American romance novelist.

 Born in Burlington, Iowa, she was the daughter of a local teacher and high school principal.

 The family moved to New York City in 1918, and "Helen Ricker" changed her name to "Elswyth Thane".

 She began working as a freelance writer in the 20s, and became a newspaper writer and a Hollywood screenwriter.

 Her first novel, "Riders of the Wind", was published in 1926.

 Her novel, "The Tudor Wench", about Elizabeth I of England, was made into a play.

 She was a collector of scarves.
</item 4>

<item 5>
The City Heiress
The City Heiress is a play by Aphra Behn produced in 1682.

 The play conforms to the general rules of Restoration comedy, but it also keeps Behn's own highly Royalist political point of view.
</item 5>

<item 6>
Lillian Hellman
Lillian Florence Hellman (June 20, 1905 – June 30, 1984) was an American dramatist and screenwriter known for her success as a playwright on Broadway, as well
as her left-wing sympathies and political activism.

 She was blacklisted after her appearance before the House Committee on Un-American Activities (HUAC) at the height of the anti-communist campaigns of 1947–52.

 Although she continued to work on Broadway in the 1950s, her blacklisting by the American film industry caused a drop in her income.

 Many praised Hellman for refusing to answer questions by HUAC, but others believed, despite her denial, that she had belonged to the Communist Party.
</item 6>

<item 7>
Sheila Levine Is Dead and Living in New York
Sheila Levine Is Dead and Living in New York is a 1975 film directed by Sidney J. Furie, about a shy young woman who moves to New York City and falls in love
with the boyfriend of her extroverted roommate.

 The film was co-written by Kenny Solms and Gail Parent, and based on her novel.

 The film was shot on location in New York City.
</item 7>

<item 8>
The Autumn Garden
The Autumn Garden is a 1951 play by Lillian Hellman.

 The play is set in September, 1949 in a summer home in a resort on the Gulf of Mexico, about 100 miles from New Orleans.

 The play is a study of the defeats, disappointments and diminished expectations of people reaching middle age.

 For inspiration, Hellman drew on her memories of her time in her aunts' boardinghouse.

 Dashiell Hammett, who had been Hellman's lover for 20 years, helped her write the play and received 15 percent of the royalties.

 Of all Hellman's plays it was her favorite.
</item 8>

<item 9>
The Heiress (1786 play)
The Heiress is a comedy play by the British playwright and soldier John Burgoyne.

 The play debuted at the Drury Lane Theatre on 14 January 1786.

 It concerns the engagement of Lord Gayville to Miss Alscrip, a fashionable woman he believes to be an heiress.

 Gayville later discovers that the woman who really stands to inherit the fortune is his true love Miss Clifford.

 The play was an enormous success, running for 31 performances in its initial season and being revived again the following year.

 The play was initially anonymous, but Burgoyne was soon widely reported to be the author and he acknowledged this after the play's debut.
</item 9>

<item 10>
Lillian Hellman: An Imperious Life
Lillian Hellman: An Imperious Life is a 2014 book by Dorothy Gallagher.

 It is a critical biography of the American playwright and writer Lillian Hellman.
</item 10>
=== END ITEM id=30e8b393c1c2a448 ===

=== ITEM id=1d766ca5e8e90b11 ===
Gold sentences and facts:
Sentence 0:  He is best known for the roles of Dan Rydell on "Sports Night;" Will Gardner on "The Good Wife", which earned him two Primetime Emmy Award nominations; and his early work as Knox Overstreet in "Dead Poets Society".
  Fact 0.0: The actor referred to as 'He' is best known for the role of Dan Rydell on "Sports Night."
  Fact 0.1: The actor referred to as 'He' is best known for the role of Will Gardner on "The Good Wife."
  Fact 0.2: The role of Will Gardner on "The Good Wife" earned the actor two Primetime Emmy Award nominations.
  Fact 0.3: The actor referred to as 'He' is best known for his early work as Knox Overstreet in "Dead Poets Society."
Sentence 1: Dead Poets Society is a 1989 American drama film directed by Peter Weir, written by Tom Schulman, and starring Robin Williams.
  Fact 1.0: Dead Poets Society is a 1989 American drama film.
  Fact 1.1: Dead Poets Society was directed by Peter Weir.
  Fact 1.2: Dead Poets Society was written by Tom Schulman.
  Fact 1.3: Dead Poets Society stars Robin Williams.
Sentence 2:  Set in 1959 at the fictional elite conservative Vermont boarding school Welton Academy, it tells the story of an English teacher who inspires his students through his teaching of poetry.
  Fact 2.0: Dead Poets Society is set in 1959.
  Fact 2.1: Dead Poets Society is set at the fictional elite conservative Vermont boarding school Welton Academy.
  Fact 2.2: Dead Poets Society tells the story of an English teacher.
  Fact 2.3: The English teacher in Dead Poets Society inspires his students through his teaching of poetry.

Retrieved items:
<item 1>
Dead Poets Society
Dead Poets Society is a 1989 American drama film directed by Peter Weir, written by Tom Schulman, and starring Robin Williams.

 Set in 1959 at the fictional elite conservative Vermont boarding school Welton Academy, it tells the story of an English teacher who inspires his students
through his teaching of poetry.
</item 1>

<item 2>
The Night Before (1988 film)
The Night Before is a 1988 film starring Keanu Reeves and Lori Loughlin.

 Reeves plays Winston Connelly, the so-called high school nerd and vice president of the astronomy club.

 Loughlin plays Tara Mitchell, the pretty and popular head cheerleader who also happens to be the local police chief's daughter.

 The Tagline was: "You lost your father's car, sold your prom date and a guy called 'Tito' wants you dead.

 It's a date that's the time of your life."

 It was filmed and set entirely in Los Angeles, California.
</item 2>

<item 3>
License to Wed
License to Wed is a 2007 American romantic comedy film starring Robin Williams, Mandy Moore and John Krasinski, and directed by Ken Kwapis.

 The film was released in theaters on July 3, 2007.
</item 3>

<item 4>
Josh Charles
Joshua Aaron "Josh" Charles (born September 15, 1971) is an American stage, film and television actor.

 He is best known for the roles of Dan Rydell on "Sports Night;" Will Gardner on "The Good Wife", which earned him two Primetime Emmy Award nominations; and his
early work as Knox Overstreet in "Dead Poets Society".
</item 4>

<item 5>
Old Boyfriends
Old Boyfriends is a 1979 American drama film directed by Joan Tewkesbury and written by Paul Schrader and Leonard Schrader.

 The film stars Talia Shire, Richard Jordan, Keith Carradine, John Belushi, John Houseman and Buck Henry.

 The film was released on April 13, 1979, by Embassy Pictures.
</item 5>

<item 6>
Dunston Checks In
Dunston Checks In is a 1996 American family comedy film starring Eric Lloyd, Graham Sack, Jason Alexander, Faye Dunaway, Rupert Everett, Paul Reubens, Glenn
Shadix, and introducing Sam the Orangutan as Dunston.

 It was written by John Hopkins and Bruce Graham and directed by Ken Kwapis.
</item 6>

<item 7>
Remember the Titans
Remember the Titans is a 2000 American sports film produced by Jerry Bruckheimer and directed by Boaz Yakin.

 The screenplay, written by Gregory Allen Howard, is based on the true story of African-American coach Herman Boone, portrayed by Denzel Washington, and his
attempt to integrate the T. C. Williams High School football team in Alexandria, Virginia, in 1971.

 Will Patton portrays Bill Yoast, Boone's assistant coach.

 Real-life athletes Gerry Bertier and Julius Campbell are portrayed by Ryan Hurst and Wood Harris, respectively.
</item 7>

<item 8>
Robin Chalk
Robin Chalk (born 1981) is an English actor, best known for his role as Neil Kellerman in the West End production of "Dirty Dancing" and his work on British
science-fiction film "Moon" directed by Duncan Jones.
</item 8>

<item 9>
The Boy with a Thorn in His Side
The Boy with a Thorn in His Side is a 2005 comedy film, produced and directed by Mark Jeavonsa as his feature film debut, and starring Alec Sedgley as Billy
Heinlickburger.

 Its title is almost the same as a song by The Smiths.
</item 9>

<item 10>
Reunion (1989 film)
The story is centred on the "enchanted friendship" of two teenagers in 1933 Germany.

 Hans Strauss (Christien Anholt) is the son of a Jewish doctor and Konradin Von Lohenburg (Samuel West) is from an aristocratic family.

 The background is the rise of Nazism.

 Jason Robards plays the older Hans in the 1970s as he prepares to travel to Germany for the first time since the 1930s.

 The film was shot on location in Berlin, New York and Stuttgart.

 "Reunion" was nominated for a Golden Palm at the 1989 Cannes Film Festival.
</item 10>
=== END ITEM id=1d766ca5e8e90b11 ===

=== ITEM id=a4008bf80c36c75a ===
Gold sentences and facts:
Sentence 0: Ranger Rick, originally Ranger Rick's Nature Magazine, is a children’s nature magazine that is published by the United States National Wildlife Federation.
  Fact 0.0: Ranger Rick was originally called Ranger Rick's Nature Magazine
  Fact 0.1: Ranger Rick is a children's nature magazine
  Fact 0.2: Ranger Rick is published by the United States National Wildlife Federation
Sentence 1: TENNIS is an American sports magazine owned by the Sinclair Broadcast Group.
  Fact 1.0: TENNIS is an American sports magazine
  Fact 1.1: TENNIS is owned by the Sinclair Broadcast Group

Retrieved items:
<item 1>
André Laguerre
Marc André Laguerre (February 21, 1915 – January 18, 1979) was a journalist and magazine editor, best known as the managing editor of "Sports Illustrated" from
1960 to 1974, during which time he oversaw the growth in the magazine from a niche publication to become the industry leader in weekly sports magazines.

 It was under his leadership that the annual "Sports Illustrated Swimsuit Issue" was first published.

 When he retired in 1974, he had been managing editor of the magazine for 704 issues, then a record among magazines published by Time, Inc., SI's parent
company.
</item 1>

<item 2>
Ranger Rick
Ranger Rick, originally Ranger Rick's Nature Magazine, is a children’s nature magazine that is published by the United States National Wildlife Federation.

 The magazine offers feature articles and activities for children, ages 7 and up, in order to spark their interest in the outdoors and become more actively
involved in the environment.

 The magazine's primary intention is to instill a passion for nature and promote activity outdoors.

 NWF also publishes two companion magazines, "Ranger Rick Jr.", which is aimed at ages 4–7, and "Ranger Rick Cub", which is aimed at kids 0–4 years old.)
</item 2>

<item 3>
Tennis (magazine)
TENNIS is an American sports magazine owned by the Sinclair Broadcast Group.

 There are eight issues per year covering the world of tennis.

 The "tennis.com" website is an offshoot of the magazine and part of the same media company.
</item 3>

<item 4>
Rap-Up
Rap-Up is a magazine launched in 2001 by founder Devin Lazerine.

 The publication was originally a website devoted to hip hop, until Lazerine decided to pitch the possibility of a magazine to several publishers.

 The magazine is focused on the hip hop and R&B aspect of the music industry, and predominantly features interviews with artists, actors and other entertainers.

 Issued quarterly, the magazine's target audience was the 14 to 28 demographic, although the demographic has become older over time.
</item 4>

<item 5>
Bicycling (magazine)
Bicycling is a cycling brand published by Rodale, Inc. in Emmaus, Pennsylvania.

 "Bicycling" claims to be the world’s largest cycling magazine.
</item 5>

<item 6>
Memphis Sport
Memphis Sport is a sports and fitness magazine featuring articles on local and regional teams, players and events.

 In addition, "Memphis Sport" focuses on health and fitness for an active lifestyle.

 The magazine debuted in Memphis, Tennessee in July 2006.

 Local franchises covered include the University of Memphis Tigers, Memphis Redbirds, Mississippi Riverkings and Tennessee Titans.
</item 6>

<item 7>
Golf Digest
Golf Digest is a monthly golf magazine published by Condé Nast Publications in the United States.

 It is a generalist golf publication covering recreational golf and men's and women's competitive golf.

 Condé Nast Publications also publishes the more specialized "Golf for Women", "Golf World" and "Golf World Business".

 The magazine started in 1950, and was sold to The New York Times Company in 1969.

 The Times company sold their magazine division to Condé Nast in 2001.

 The headquarters of "Golf Digest" is in Des Moines, Iowa.
</item 7>

<item 8>
El Gráfico
El Gráfico is a classic monthly Argentine sports magazine.

 The magazine was originally published as a weekly newspaper in May 1919, and then turned to a sports magazine exclusively.

 It has been scheduled monthly from 2002.

 "El Gráfico" is one of the most regarded sports magazines not only in Argentina but in Latin America.

 The magazine was nicknamed "La Biblia del deporte" ("The Bible of sports") due to its chronicles, notable journalists and collaborators and its photography.
</item 8>

<item 9>
Lorin Thompson
Lorin Thompson, Lorin Hartwell Jr Thompson, (1911–1997), was a muralist, artist, and creator and illustrator of the character "Ranger Rick" for the National
Wildlife Federation's children's magazine, Ranger Rick.
</item 9>

<item 10>
Guitar World
Guitar World is a monthly music magazine devoted to guitarists, published since July 1980.

 It contains original interviews, album and gear reviews, and guitar and bass tablature of approximately five songs each month.

 The magazine is published 13 times per year (12 monthly issues and a holiday issue).

 Formerly owned by Harris Publications, Future US bought the magazine in 2003.

 In 2012, NewBay Media bought the Music division of Future US.

 The latter company also published a spin-off title, Guitar Legends, each issue of which typically combined past articles from "Guitar World" under a specific
theme.
</item 10>
=== END ITEM id=a4008bf80c36c75a ===

=== ITEM id=8acdbd1e76de9695 ===
Gold sentences and facts:
Sentence 0:  The album was Prince's third to be released that year (along with "Chaos and Disorder" and the soundtrack album of the Spike Lee movie "Girl 6"), which made 1996 one of the most prolific years for material released by Prince.
  Fact 0.0: The album was Prince's third album released in 1996.
  Fact 0.1: One of Prince's other 1996 releases was "Chaos and Disorder".
  Fact 0.2: One of Prince's other 1996 releases was the soundtrack album of the Spike Lee movie "Girl 6".
  Fact 0.3: 1996 was one of the most prolific years for material released by Prince.
Sentence 1:  His production company, 40 Acres and a Mule Filmworks, has produced over 35 films since 1983.
  Fact 1.0: Prince's production company is 40 Acres and a Mule Filmworks.
  Fact 1.1: 40 Acres and a Mule Filmworks has produced over 35 films since 1983.

Retrieved items:
<item 1>
Chris Gorak
Chris Gorak is an American film director who started as an art director and production designer.

 He directed the 2006 film "Right at Your Door" and the 2011 film "The Darkest Hour".

 Gorak began working in the film industry in the 1990s as an art director.

 In the 2000s, he worked on several films as production designer before becoming a film director.
</item 1>

<item 2>
Still Kicking (film)
Still Kicking: Six Artistic Women of Project Arts & Longevity is a 2006 32-minute documentary film by Pacific Grove filmmaker Greg Young, featuring six Bay Area
women role models over 85 years old who remained artistically active.

 The catalyst for Young's film was Amy Gorman and Frances Kandl's Project Arts & Longevity through which they were exploring the link between longevity and
artistic vitality.

 Along with the film the joint project resulted in a book entitled Aging Artfully.
</item 2>

<item 3>
Mark Lee Ping-bing
Mark Lee Ping-bing (李屏賓; born August 8, 1954) is a Taiwanese cinematographer, photographer and author with over 70 films and 21 international awards to his
credit including 2 Glory Of The Country Awards from the Government Information Office of Taiwan and the president of Taiwan's Light Of The Cinema Award.

 Lee began his film career in 1977 and in 1985 he started his prolific collaboration with Taiwanese filmmaker Hou Hsiao-hsien.

 Known best for his use of natural lighting utilizing real film and graceful camera movement, Lee received the Grand Technical Prize at the Cannes Film Festival
in 2000 for "In the Mood for Love".

 A member of the Academy of Motion Picture Arts and Sciences, Lee was honored with nominations by the American Society of Cinematographers for its 2014 First
Annual Spotlight Award for Best Cinematography for his work on the 2012 film "Renoir" and by the French Academy of Cinema Arts for a Cesar Award for Best
Cinematography in 2014 also for the film "Renoir".
</item 3>

<item 4>
Spike Lee
Shelton Jackson "Spike" Lee (born March 20, 1957) is an American film director, producer, writer, and actor.

 His production company, 40 Acres and a Mule Filmworks, has produced over 35 films since 1983.
</item 4>

<item 5>
Cannibal boom
Only four other cannibal films were made after 1981 until the fad's conclusion in 1988 with Antonio Climati's "Natura Contro".
</item 5>

<item 6>
Mel Gibson filmography
Mel Gibson, AO, is an American actor, director, producer and screenwriter who made his acting debut on the Australian television drama series "The Sullivans"
(1976– 1983).

 While a student at the National Institute of Dramatic Art in Sydney, he was given an uncredited role in "I Never Promised You a Rose Garden" and subsequently
appeared as a leading actor in the micro budget surf drama "Summer City" (both in 1977).

 Gibson rose to prominence during the Australian New Wave cinema movement in the early 1980s, having appeared in his breakthrough role in George Miller's
dystopian action film "Mad Max" (1979), portraying the eponymous hero.

 He reprised the role in its sequels, "Mad Max 2" (1981) and "Mad Max Beyond Thunderdome" (1985).

 He appeared in Peter Weir's war drama "Gallipoli" (1981) and the romantic drama "The Year of Living Dangerously" (1982).
</item 6>

<item 7>
List of accolades received by The Sixth Sense
"The Sixth Sense" is an American supernatural thriller film, written and directed by M. Night Shyamalan.

 The film was released on August 6, 1999, grossing over $26,600,000 on its opening weekend and ranking first place at the box office.

 Overall the film grossed over $293,500,000 domestically and $672,800,000 worldwide, which is approximately 16.8 times its budget of $40 million.

 "The Sixth Sense" was well received by critics, with an approval rating of 85% from review aggregator Rotten Tomatoes.
</item 7>

<item 8>
The Human Centipede 2 (Full Sequence)
The Human Centipede 2 (Full Sequence) is a 2011 Dutch black and white exploitation-horror film written, directed, and co-produced by Dutch filmmaker Tom Six.

 The sequel to Six's 2009 film "The Human Centipede (First Sequence)", the film stars Laurence R. Harvey as a mentally impaired English man who watches and
becomes obsessed with the first "Human Centipede" film, and decides to make his own "centipede" consisting of twelve people, including Ashlynn Yennie, an
actress from the first film.
</item 8>

<item 9>
Girl 6
Girl 6 is a 1996 American comedy-drama film by director Spike Lee about a phone sex operator.

 Theresa Randle played the title character, and playwright Suzan-Lori Parks wrote the screenplay.

 The soundtrack is composed entirely of songs written by Prince.

 The film was screened in the Un Certain Regard section at the 1996 Cannes Film Festival.

 Directors Quentin Tarantino and Ron Silver make cameo appearances as film directors at a pair of interesting auditions.

 It is the first film directed by Lee in which he did not write the screenplay.
</item 9>

<item 10>
Dan Ouellette
Giger and Beksinski, and he has been featured in magazines internationally.

 David Bowie commented while looking at Dan's art that he has noticed a strong influence of science fiction on contemporary art.

 Giger saw a different aspect, saying simply "Very bony."

 He grew to adore the cinema of Fellini and later to marvel at Lynch's "Eraserhead".

 Rather than pursue a formal education in the fine arts he chose to study the craft of film making and after college he became a production designer working on
feature films in New York City starting with his work as a production designer for Hal Hartley in 1990 with Trust and then, in 1992, with Simple Men.
</item 10>
=== END ITEM id=8acdbd1e76de9695 ===

=== ITEM id=cfdaddf4f0c1d34a ===
Gold sentences and facts:
Sentence 0:  He received an excellent education in both Arabic and Greek, having studied Arabic grammar with the famous poet Abul ʿAla Al-Maʿarri.
  Fact 0.0: He received an excellent education in Arabic.
  Fact 0.1: He received an excellent education in Greek.
  Fact 0.2: He studied Arabic grammar with the famous poet Abul ʿAla Al-Maʿarri.
Sentence 1: Abul ʿAla Al-Maʿarri (Arabic أبو العلاء المعري "Abū al-ʿAlāʾ al-Maʿarrī ", full name أبو العلاء أحمد بن عبد الله بن سليمان التنوخي المعري "Abū al-ʿAlāʾ Aḥmad ibn ʿAbd Allāh ibn Sulaimān al-Tanūẖī al-Maʿarrī "; December 973 – May 1057) was a blind Arab philosopher, poet, and writer.
  Fact 1.0: Abul ʿAla Al-Maʿarri's Arabic name is أبو العلاء المعري (Abū al-ʿAlāʾ al-Maʿarrī).
  Fact 1.1: Abul ʿAla Al-Maʿarri's full name is أبو العلاء أحمد بن عبد الله بن سليمان التنوخي المعري (Abū al-ʿAlāʾ Aḥmad ibn ʿAbd Allāh ibn Sulaimān al-Tanūẖī al-Maʿarrī).
  Fact 1.2: Abul ʿAla Al-Maʿarri was born in December 973.
  Fact 1.3: Abul ʿAla Al-Maʿarri died in May 1057.
  Fact 1.4: Abul ʿAla Al-Maʿarri was blind.
  Fact 1.5: Abul ʿAla Al-Maʿarri was an Arab philosopher.
  Fact 1.6: Abul ʿAla Al-Maʿarri was an Arab poet.
  Fact 1.7: Abul ʿAla Al-Maʿarri was an Arab writer.

Retrieved items:
<item 1>
Abd Allah ibn Abbas
Abd Allah ibn Abbas (Arabic: عبد الله ابن عباس‎ ‎ ) or ′Abd Allah ibn al-′Abbas otherwise called (Ibn Abbas; Al-Habr; Al-Bahr; The Doctor; The Sea) was born c.
619 CE.

 He was the son of Al-‘Abbas ibn ‘Abd al-Muttalib, an uncle of the Islamic prophet Muhammad, and a nephew of the Maymunah bint al-Harith, who later became
Muhammad's wife.

 He was one of Muhammad's cousins and one of the early Qur'an scholars.
</item 1>

<item 2>
Ibn al-Khatib
Lisan ad-Din ibn al-Khatib (Born 16 November 1313, Loja– died 1374, Fes, Morocco) (Full name Muhammad ibn Abd Allah ibn Said ibn Ali ibn Ahmad al-Salmani) was a
Moroccan Arab polymath poet, writer, historian, philosopher, physician and politician from Emirate of Granada.

 Some of his poems decorate the walls of the Alhambra in Granada.
</item 2>

<item 3>
Al-Maʿarri
Abul ʿAla Al-Maʿarri (Arabic أبو العلاء المعري "Abū al-ʿAlāʾ al-Maʿarrī ", full name أبو العلاء أحمد بن عبد الله بن سليمان التنوخي المعري "Abū al-ʿAlāʾ Aḥmad
ibn ʿAbd Allāh ibn Sulaimān al-Tanūẖī al-Maʿarrī "; December 973 – May 1057) was a blind Arab philosopher, poet, and writer.

 Al-Maʿarri held and expressed an irreligious world view which was met with controversy, but in spite of it, he is regarded as one of the greatest classical
Arabic poets.
</item 3>

<item 4>
Al-Mutanabbi
Al-Mutanabbi had a great pride in himself through his poetry.

 Among the topics he discussed were courage, the philosophy of life, and the description of battles.

 Many of his poems were and still are widely spread in today's Arab world and are considered to be proverbial.
</item 4>

<item 5>
Al-Mutanabbi
Abu at-Tayyib Ahmad bin Al-Husayn al-Mutanabbi al-Kindi (Arabic: أبو الطيب أحمد بن الحسين المتنبّي الكندي‎ , "Abū ṭ-Ṭayyib ʾAḥmad bin al-Ḥusayn al-Muṫanabbī
al-Kindī " ‎ ) (915 – 23 September 965 CE) was an Arab poet.

 He is considered as one of the greatest poets in the Arabic language and is the most prominent and most influential poet in the Arab world and much of his work
has been translated into over 20 languages worldwide.

 Much of his poetry revolves around praising the kings he visited during his lifetime.

 Some consider his 326 poems to be a great representation of his life story.

 He started writing poetry when he was nine years old.

 He is well known for his sharp intelligence and wittiness.

 Al-Mutanabbi had a great pride in himself through his poetry.
</item 5>

<item 6>
Da'ud Abu al-Fadl
Da'ud Abu al-Fadl (1161–1242) was a Karaite Jewish physician who lived in the Ayyubid Egypt in the twelfth century CE.

 He born in Cairo in 1161 and died there about 1242.

 Having studied medicine under the Jewish physician Hibat Allah ibn Jami, and under Abu al-Fafa'il ibn Naqid, he became the court physician of the sultan
al-Malik al-'Adil Abu Bakr ibn Ayyub, the brother and successor of Saladin.

 He was also chief professor at the al-Nasiri Hospital at Cairo, where he had a great many pupils, among them being the historian ibn Abi Usaibiyyah.

 The latter declared that Abu al-Fadl was the most skillful physician of the time and that his success in curing the sick was miraculous.

 Abu al-Fadl was the author of an Arabic pharmacopoeia in twelve chapters, entitled "Aḳrabadhin", and treating chiefly of antidotes.
</item 6>

<item 7>
Antarah ibn Shaddad
Antarah ibn Shaddad (Arabic: عنترة بن شداد العبسي‎ ‎ , "ʿAntarah ibn Shaddād al-ʿAbsī";  525–608), also known as ʿAntar, was a pre-Islamic Arab knight and poet,
famous for both his poetry and his adventurous life.

 His chief poem forms part of the "Mu'allaqāt", the collection of seven "hanging odes" legendarily said to have been suspended in the Kaaba.

 The account of his life forms the basis of a long and extravagant romance.
</item 7>

<item 8>
Abd Allah ibn al-Fadl
Little is known of his life, apart from what can be gleaned from manuscripts of his texts.

 He was a deacon and the grandson of a bishop.

 He received an excellent education in both Arabic and Greek, having studied Arabic grammar with the famous poet Abul ʿAla Al-Maʿarri.

 Several of his works and translations were commissioned by notables from Antioch and neighbouring cities in Muslim territory.
</item 8>

<item 9>
Abdal
Abdāl ("lit": substitutes, but which can also mean "generous" ["karīm"] and "noble" ["sharīf"]) is a term used in Islamic metaphysics and Islamic mysticism,
both Sunni and Shiite, to refer to a particularly important group of God's saints.

 In the tradition of Sunni Islam in particular, the concept attained an especially important position in the writings of the Sunni mystics and theologians,
whence it appears in the works of Sunni authorities as diverse as Abu Talib al-Makki (d. 956), Ali Hujwiri (d. 1072), Ibn Asakir (d. 1076), Khwaja Abdullah
Ansari (d. 1088), and Ibn Khaldun (d. 1406).
</item 9>

<item 10>
Shaykh Muhammad ibn Thaleb
Muhammad ibn Thalib ibn Abd Allah ibn Ni`mat Allah ibn Sadr ad-Din ibn Shaykh Baha' ad-Din ash-Shirazi (Arabic: محمد ابن طالب عبدالله ابن نعمات الله ابن صدر
الدين ابن شيخ بهاءالدين الشيرازي‎ ‎ ) was a 15th-century Persian physician from Shiraz, Iran.
</item 10>
=== END ITEM id=cfdaddf4f0c1d34a ===

=== ITEM id=1249253421816b8b ===
Gold sentences and facts:
Sentence 0: The Livesey Hall War Memorial commemorates the fallen of World War I and World War II who had been employed by the South Suburban Gas Company of London.
  Fact 0.0: The Livesey Hall War Memorial commemorates the fallen of World War I.
  Fact 0.1: The Livesey Hall War Memorial commemorates the fallen of World War II.
  Fact 0.2: The commemorated fallen had been employed by the South Suburban Gas Company of London.
Sentence 1:  Over 60 million people were killed, which was about 3% of the 1940 world population (est. 2.3 billion).
  Fact 1.0: Over 60 million people were killed.
  Fact 1.1: The deaths were about 3% of the 1940 world population.
  Fact 1.2: The estimated 1940 world population was 2.3 billion.

Retrieved items:
<item 1>
Tower Hill Memorial
The Tower Hill Memorial is a Commonwealth War Graves Commission war memorial on the south side of Trinity Square Gardens, in London, England.

 The memorial commemorates those from the Merchant Navy and fishing fleets who died during both world wars and have "no grave but the sea".

 The memorial was designed by Edwin Lutyens with sculpture work by William Reid Dick, the Second World War extension was designed by Edward Maufe with sculpture
work by Charles Wheeler.
</item 1>

<item 2>
Rainham War Memorial
The Rainham War Memorial commemorates soldiers killed in both World Wars as well as civilian casualties of World War II.
</item 2>

<item 3>
Wood Green War Memorial
The Wood Green War Memorial is located on the High Road, Wood Green, in the London Borough of Haringey.

 It was erected in 1920 and paid for by public subscription of the people of Wood Green to remember the men of the area who died during the First World War.

 It was subsequently adapted to include the dead of the Second World War.

 It is grade II listed with Historic England.
</item 3>

<item 4>
Algernon Willis
Admiral of the Fleet Sir Algernon Usborne Willis (17 May 1889 – 12 April 1976) was a Royal Navy officer.

 He served in the First World War and saw action at the Battle of Jutland in May 1916.

 He also served in the Second World War as Commander-in-Chief, South Atlantic in which capacity he led actions against German and Japanese raiding ships.

 He continued his war service as Flag Officer commanding 3rd Battle Squadron and Second in command of the Eastern Fleet and then as Flag Officer commanding
Force H, the force which covered North African Operations, the Allied invasion of Sicily in July 1943 and then the Allied invasion of Italy in September 1943.

 He spent the final years of the war as Commander-in-Chief, Levant, in which capacity he conducted naval operations in support of the Dodecanese Campaign, and
then as Second Sea Lord, in which capacity he arranged the manpower for the campaign in the Pacific Ocean against the Imperial Japanese Navy.

 After the war he served as Commander-in-Chief, Mediterranean Fleet, in which role he was faced with unrest in Mandatory Palestine, before he became
Commander-in-Chief, Portsmouth.
</item 4>

<item 5>
Portsmouth Naval Memorial
The Portsmouth Naval Memorial, sometimes known as Southsea Naval Memorial, is a war memorial in Portsmouth, Hampshire, England, on Southsea Common beside
Clarence Esplanade, between Clarence Pier and Southsea Castle.

 The memorial commemorates approximately 25,000 British and Commonwealth sailors who were lost in the World Wars, around 10,000 sailors in the First World War
and 15,000 in the Second World War.

 The memorial features a central obelisk, with names of the dead on bronze plaques arranged around the memorial according to the year of death.
</item 5>

<item 6>
Alamein Memorial
The Alamein Memorial is a Commonwealth War Graves Commission war memorial in the El Alamein War Cemetery, El Alamein, Egypt.

 The memorial commemorates 11,866 Commonwealth forces members who died during World War II.

 The memorial was designed by Hubert Worthington and unveiled by Viscount Montgomery of Alamein on 24 October 1954.
</item 6>

<item 7>
Arras Flying Services Memorial
The Arras Flying Services Memorial Commonwealth War Graves Commission war memorial in the Faubourg d'Amiens Cemetery, Arras, France.

 The memorial commemorates nearly 1,000 airmen from forces of the Commonwealth who were killed on the Western Front during World War I and who have no known
grave.

 The memorial was designed by Edwin Lutyens, sculpted by William Reid Dick and unveiled by Hugh Trenchard, 1st Viscount Trenchard, Marshal of the Royal Air
Force on 31 July 1932.
</item 7>

<item 8>
Freetown Memorial
The Freetown Memorial is a Commonwealth War Graves Commission war memorial located outside the Secretariate Building in Freetown, Sierra Leone.

 The memorial generally commemorates Soldiers from Sierra Leone who died whilst serving with the Royal West African Frontier Force in West Africa and whose
graves are not known.

 The Freetown Memorial commemorates 1,109 First World War casualties and 247 from the Second World War.
</item 8>

<item 9>
Livesey Hall War Memorial
The Livesey Hall War Memorial commemorates the fallen of World War I and World War II who had been employed by the South Suburban Gas Company of London.

 It is also a tribute to those employees who served in the wars.

 The monument was designed and executed by British sculptor Sydney March, of the March family of artists.
</item 9>

<item 10>
World War II casualties
World War II was the deadliest military conflict in history in absolute terms of total casualties.

 Over 60 million people were killed, which was about 3% of the 1940 world population (est. 2.3 billion).

 The tables below give a detailed country-by-country count of human losses.

 World War II fatality statistics vary, with estimates of total deaths ranging from 50 million to more than 80 million.

 The higher figure of over 80 million includes deaths from war-related disease and famine.

 Civilians killed totalled 50 to 55 million, including 19 to 28 million from war-related disease and famine.

 Military deaths from all causes totalled 21 to 25 million, including deaths in captivity of about 5 million prisoners of war.
</item 10>
=== END ITEM id=1249253421816b8b ===

=== ITEM id=d4aa5bc4ab9e67f3 ===
Gold sentences and facts:
Sentence 0: Charles Thomas "Tom" Johnston (born August 15, 1948) is an American musician.
  Fact 0.0: Charles Thomas 'Tom' Johnston was born on August 15, 1948.
  Fact 0.1: Charles Thomas 'Tom' Johnston is an American musician.
  Fact 0.2: Charles Thomas Johnston is known by the name 'Tom'.
Sentence 1: Daryl Palumbo (born February 10, 1979) is an American musician, originally from Elmira, New York.
  Fact 1.0: Daryl Palumbo was born on February 10, 1979.
  Fact 1.1: Daryl Palumbo is an American musician.
  Fact 1.2: Daryl Palumbo is originally from Elmira, New York.

Retrieved items:
<item 1>
Tom Johnston (musician)
Charles Thomas "Tom" Johnston (born August 15, 1948) is an American musician.

 He is a guitarist and vocalist, known principally as a founder, guitarist, lead vocalist and songwriter for the rock group The Doobie Brothers, as well as for
his own solo career.

 His off-and-on career with The Doobie Brothers spans more than 40 years, during which time the band exhibited numerous successful styles.
</item 1>

<item 2>
Head Automatica
Head Automatica is an American rock band from Long Island, New York fronted by Daryl Palumbo (also of Glassjaw).
</item 2>

<item 3>
Bob Thomas (actor)
Bob Thomas (born March 1, 1954 in Appalachia, Virginia ) is a radio personality, actor, and writer.

 He was one of the top radio announcers in Knoxville, Tennessee for 25 years.

 As an actor, he has appeared in many films and hundreds of commercials.

 He wrote two episodes of the popular TV show "Lizzie McGuire".

 In 2004, his voice was heard as "Slammin' Sammy" in the film "Friday Night Lights".
</item 3>

<item 4>
Phil Driscoll
Phil Driscoll (born November 9, 1947) is a trumpeter, singer, composer, and producer.

 He performs in varying music genres and styles which include rock and roll, rhythm and blues, and patriotic music, and is best known for his work in Christian
music and his longterm Christian ministry.

 In 1985, Driscoll won the Grammy Award for Best Gospel Performance – Duo/Group for a duet with Debby Boone on "Keep the Flame Burning", and he has been
nominated for three additional Grammys, two for Best Gospel Performance – Male and one for Best Gospel/Pop Album.

 He has also won three Dove Awards for his music, and the 1999 Christian Country Music Association Award for Best Musician.

 In 2006, Driscoll was found guilty on 2 counts of tax evasion and one count of conspiracy, and was sentenced to serve one year in Federal prison, beginning on
March 14, 2007.
</item 4>

<item 5>
Tom Johnston
Tom Johnston, Thomas Johnston or Tommy Johnston may refer to:
</item 5>

<item 6>
James Dooley (composer)
James Michael "Jim" Dooley (born August 22, 1976 in New York City, New York) is an American film score composer.

 Dooley studied music at New York University, majoring in music composition.

 After finishing the university he moved to Los Angeles, where he studied music with prolific film score composers Christopher Young, Elmer Bernstein and
Leonard Rosenman.

 In 1999, he started working for Hans Zimmer as his chief technical assistant.

 He works in Santa Monica, in Hans Zimmer's film music studio Remote Control Productions (formerly "Media Ventures").

 He composed, arranged, and orchestrated music for films like "" and "The Da Vinci Code".

 He also composed music for "inFAMOUS 2", the Epic Mickey series and worked with Celldweller and Tarja Turunen.

 He released his debut album, "Veiled Nation", in 2013.
</item 6>

<item 7>
Graph query row:
person: Daryl Palumbo
profession: musician
</item 7>

<item 8>
Bullmoose
Bullmoose (aka Zach Dobbins) is a Canadian rock band formed in Elgin, Ontario in 1996.

 Three of the original members continue in the band, which today is based in Montreal.

 They include twin brothers Jeff Cowan (drums/vocals) and Seamus Cowan (bass/vocals) from Westport, Ontario and Eric Lawrance (guitar/vocals) from Delta,
Ontario, who launched the band when they attended school at Montreal.

 The band performs in the rock music genre, citing experimental influences of the late 1960s and early 1970s.
</item 8>

<item 9>
Graph query row:
person: Tom Johnston
profession: ['musician', 'guitarist', 'vocalist', 'founder', 'lead vocalist', 'songwriter']
</item 9>

<item 10>
Daryl Palumbo
Daryl Palumbo (born February 10, 1979) is an American musician, originally from Elmira, New York.

 He is the frontman of the bands Glassjaw, Head Automatica and Color Film.

 As a youth he was a member of the Long Island straight edge band "XbustedX."

 In 1993 he met guitarist Justin Beck, who would become his good friend.

 Together they formed Glassjaw.
</item 10>
=== END ITEM id=d4aa5bc4ab9e67f3 ===

=== ITEM id=a02af0cd06e13c12 ===
Gold sentences and facts:
Sentence 0: Colocasia is a genus of flowering plants in the family Araceae, native to southeastern Asia and the Indian subcontinent.
  Fact 0.0: Colocasia is a genus of flowering plants.
  Fact 0.1: Colocasia is in the family Araceae.
  Fact 0.2: Colocasia is native to southeastern Asia.
  Fact 0.3: Colocasia is native to the Indian subcontinent.
Sentence 1: The genus Coronilla contains about 20 species of flowering plants native to Europe and North Africa.
  Fact 1.0: The genus Coronilla contains about 20 species.
  Fact 1.1: The Coronilla species are flowering plants.
  Fact 1.2: The Coronilla species are native to Europe.
  Fact 1.3: The Coronilla species are native to North Africa.

Retrieved items:
<item 1>
Dieffenbachia
Dieffenbachia is a genus of tropical flowering plants in the family Araceae.

 It is native to the New World Tropics from Mexico and the West Indies south to Argentina.

 Some species are widely cultivated as ornamental plants, especially as houseplants, and have become naturalized on a few tropical islands.
</item 1>

<item 2>
Coronilla
The genus Coronilla contains about 20 species of flowering plants native to Europe and North Africa.
</item 2>

<item 3>
Coris monspeliensis
Coris monspeliensis has erect or ascending stems which are woody at the base and densely covered with foliage.

 The leathery linear leaves, which grow up to 20 mm long are alternate and stalkless.

 They may be hairless or hairy and the highest leaves often have small teeth.

 The flowers with 5 unequal petals, each deeply lobed are borne in short dense clusters and vary in color from pink to blue.

 Each flower measures up to 12 mm across and has narrow, widely spread petals.

 The calyx is bell shaped and has up to 20 red or black spiny teeth.

 The spherical capsule is up to 2 mm in diameter.
</item 3>

<item 4>
Caryophyllales
Caryophyllales ( ) is an order of flowering plants that includes the cacti, carnations, amaranths, ice plants, beets, and many carnivorous plants.

 Many members are succulent, having fleshy stems or leaves.
</item 4>

<item 5>
Strobilanthes
Strobilanthes is a genus of about 350 species of flowering plants in the family Acanthaceae, mostly native to tropical Asia and Madagascar, but with a few
species extending north into temperate regions of Asia.

 Many species are cultivated for their 2-lipped, hooded flowers in shades of blue, pink, white and purple.

 Most are frost-tender and require protection in frost-prone areas.
</item 5>

<item 6>
Conicosia
Conicosia is a genus of succulent plants in the ice plant family native to southern Africa.

 They are known commonly as narrow-leafed ice plants.

 These are relatively short-lived perennials with underground stems and tentacle-shaped, dull-pointed triangular leaves.

 They bear large tubular flowers often exceeding 10 centimeters in width, with up to 250 fringelike petals arranged in a ring around a center with hundreds of
stamens.

 The fruit is a capsule which opens when it gets wet, slowly releasing the hundreds of tiny seeds as they fall out of its drying flesh.
</item 6>

<item 7>
Conophytum
Conophytum is a genus of South African and Namibian succulent plants that belong to the Aizoaceae family.

 The name is derived from the Latin “conus” (cone) and Greek “phytum” (plant).

 The plants are also known as knopies (buttons), waterblasies (water blisters), sphaeroids, conos, cone plants, dumplings, or button plants.
</item 7>

<item 8>
Colocasia
Colocasia is a genus of flowering plants in the family Araceae, native to southeastern Asia and the Indian subcontinent.

 Some species are widely cultivated and naturalized in other tropical and subtropical regions.

 Common names include tarul, karkala ko ganu, elephant-ear, taro, cocoyam, dasheen, chembu, champadhumpa, shavige gadde, and eddoe.

 Elephant-ear and cocoyam are also used for some other large-leaved genera in the Araceae, notably "Xanthosoma" and "Caladium".

 The generic name is derived from the ancient Greek word "kolokasion", which in the Greek botanist Dioscorides (1st century AD) meant the edible roots of both
"Colocasia esculenta" and "Nelumbo nucifera".

 It is thought that the edible roots of "Colocasia esculenta" have been cultivated in Asia for more than ten thousand years.

 The species "Colocasia esculenta" is an invasive species in wetlands along the American Gulf coast, where it threatens to displace native wetland plants.
</item 8>

<item 9>
Caladium
Caladium is a genus of flowering plants in the family Araceae.

 They are often known by the common name elephant ear (which they share with the closely related genera "Alocasia", "Colocasia", and "Xanthosoma"), Heart of
Jesus, and Angel Wings.

 There are over 1000 named cultivars of "Caladium bicolor" from the original South American plant.
</item 9>

<item 10>
Xanthosoma
Xanthosoma is a genus of flowering plants in the arum family, Araceae.

 The genus is native to tropical America but widely cultivated and naturalized in other tropical regions.

 Several are grown for their starchy corms, an important food staple of tropical regions, known variously as "malanga", "otoy", "otoe", cocoyam (or new
cocoyam), "tannia", "tannier", "yautía", "macabo", "ocumo", "macal", "taioba", "dasheen", "quequisque", "ʻ ape" and (in Papua New Guinea) as Singapore taro
("taro kongkong").

 Many other species (including especially "X. roseum") are used as ornamental plants, and in popular horticultural literature are known as ‘ape or elephant ear
(from the purported resemblance of the leaf to an elephant's ear), although the latter name is sometimes also applied to members with similar appearance and
uses in the closely related genera "Caladium", "Colocasia" (i.e., taro), and "Alocasia".
</item 10>
=== END ITEM id=a02af0cd06e13c12 ===

=== ITEM id=772826af5b0c4281 ===
Gold sentences and facts:
Sentence 0:  The event was broadcast live on BBC One and was hosted by Graham Norton.
  Fact 0.0: The event was broadcast live on BBC One.
  Fact 0.1: Graham Norton hosted the event.
Sentence 1: Graham William Walker (born 4 April 1963), better known as Graham Norton, is an Irish television and radio presenter, DJ, comedian, actor, and writer.
  Fact 1.0: Graham William Walker was born on 4 April 1963.
  Fact 1.1: Graham William Walker is better known as Graham Norton.
  Fact 1.2: Graham Norton is an Irish television presenter.
  Fact 1.3: Graham Norton is an Irish radio presenter.
  Fact 1.4: Graham Norton is an Irish DJ.
  Fact 1.5: Graham Norton is an Irish comedian.
  Fact 1.6: Graham Norton is an Irish actor.
  Fact 1.7: Graham Norton is an Irish writer.

Retrieved items:
<item 1>
BAFTA Scotland
BAFTA in Scotland is the Scottish branch of the British Academy of Film and Television Arts.

 Formed in 1986, the branch holds two annual awards ceremonies recognising the achievement by performers and production staff in Scottish film, television and
video games.

 These Awards are separate from the British Academy Television Awards and British Academy Film Awards.
</item 1>

<item 2>
British Academy Children's Awards
The British Academy Children's Awards are presented in an annual award show hosted by the British Academy of Film and Television Arts (BAFTA).

 They have been awarded annually since 1996.

 Before that, children's awards were a part of the main British Academy Television Awards.
</item 2>

<item 3>
2013 Africa Magic Viewers Choice Awards
The 2013 Africa Magic Viewers Choice Awards were held on 9 March 2014 and co-hosted by Big Brother Africa presenter IK Osakioduwa and StarGist’s host, Vimbai
Mutinhiri.

 Ivie Okujaiye and Olu Jacobs were honored with the TrailBlazer and Industry Merit Awards respectively.
</item 3>

<item 4>
British Academy Television Award for Best Actress
This is a list of the British Academy Television Awards for Best Actress.

 The British Academy Television Awards began in 1955.

 The Best Actress award was initially given as an 'individual honour' without credit to a particular performance until 1969 when Wendy Craig won for her
performance in "Not in Front of the Children".

 Since 1970, nominees have been announced in addition to the winner and are listed with the winner highlighted in blue.

 The Actress category was split into Leading Actress and Supporting Actress starting in 2010.
</item 4>

<item 5>
British Academy Television Awards 1997
The 1997 British Academy Television Awards were held on 29 April at the Royal Albert Hall in London, as a joint ceremony with the British Academy Film Awards.

 To date, it is the last occasion upon which the two sets awards have been given jointly at the same event; from 1998, the Film and Television Awards were given
at separate ceremonies, for the first time since 1968.
</item 5>

<item 6>
British Academy Television Awards 1998
The 1998 British Academy Television Awards were held on 18 May at the Prince of Wales Theatre in London.

 The ceremony was broadcast on ITV, hosted by Bob Monkhouse, and it was the first occasion since 1968 that the Television Awards had been held separately from
the British Academy Film Awards, instead of as a joint ceremony.
</item 6>

<item 7>
Graham Norton
Graham William Walker (born 4 April 1963), better known as Graham Norton, is an Irish television and radio presenter, DJ, comedian, actor, and writer.

 He is a five-time BAFTA TV Award winner for his comedy chat show "The Graham Norton Show".

 Previously shown on BBC Two, it took the prestigious Friday night slot on BBC One from "The Jonathan Ross Show" in 2010.
</item 7>

<item 8>
2012 British Academy Television Awards
The 2012 British Academy Television Awards (formally known as the Arqiva British Academy Television Awards) were held on 27 May 2012 at the Royal Festival Hall
in London.
</item 8>

<item 9>
British Academy Television Awards 2009
The 2009 British Academy Television Awards were held on 26 April at the Royal Festival Hall in London.

 The event was broadcast live on BBC One and was hosted by Graham Norton.

 The nominations were announced on 24 March.

 Winners in bold.
</item 9>

<item 10>
British Academy Television Awards
The British Academy Television Awards, also known as the BAFTA Television Awards, are presented in an annual award show hosted by the British Academy of Film
and Television Arts (BAFTA).

 They have been awarded annually since 1955.

 It is the British equivalent to the Emmy Awards in the United States and Logie Awards in Australia.
</item 10>
=== END ITEM id=772826af5b0c4281 ===

=== ITEM id=c66c36f739a8fa6e ===
Gold sentences and facts:
Sentence 0: Carly Gullickson (born November 26, 1986) is a former American professional tennis player.
  Fact 0.0: Carly Gullickson was born on November 26, 1986.
  Fact 0.1: Carly Gullickson is a former American professional tennis player.
Sentence 1: Francesca Schiavone (] ; born 23 June 1980 in Milan) is an Italian tennis player who turned professional in 1998.
  Fact 1.0: Francesca Schiavone was born on 23 June 1980 in Milan.
  Fact 1.1: Francesca Schiavone is an Italian tennis player.
  Fact 1.2: Francesca Schiavone turned professional in 1998.

Retrieved items:
<item 1>
Carly Gullickson
Carly Gullickson (born November 26, 1986) is a former American professional tennis player.
</item 1>

<item 2>
2005 Challenge Bell – Doubles
Carly Gullickson and María Emilia Salerni were the defending champions, but Gullickson decided not to participate this year.

 Salerni partnered with Marion Bartoli, but withdrew from their semifinal match against Līga Dekmeijere and Ashley Harkleroad.
</item 2>

<item 3>
2006 Generali Ladies Linz – Doubles
Gisela Dulko and Květa Peschke were the defending champions, but competed this year with different partners.

 Dulko teamed up with Michaëlla Krajicek and lost in first round to Eleni Daniilidou and Jasmin Wöhr, while Peschke teamed up with Francesca Schiavone and had
to withdrew in their semifinal match (against Corina Morariu and Katarina Srebotnik) as Schiavone had her right knee injured.
</item 3>

<item 4>
Francesca Schiavone
Francesca Schiavone (] ; born 23 June 1980 in Milan) is an Italian tennis player who turned professional in 1998.

 She won the 2010 French Open singles title, becoming the first Italian woman to win a Grand Slam event in singles.

 She was also runner-up at the 2011 French Open.

 Her career high ranking is world No. 4, achieved on 31 January 2011.

 To date, Schiavone is the last one handed-backhand player to win a Grand Slam title on the women's tour.
</item 4>

<item 5>
2007 Kremlin Cup – Women's Doubles
Květa Peschke and Francesca Schiavone were the defending champions, but Schiavone chose not to participate, and only Peschke competed that year.
</item 5>

<item 6>
Nicole Strausak
Nicole Strausak (born October 21, 1971) is a Swiss curler who retired from competitive curling in March 2005.

 Amongst other awards, she received a World Junior Silver medal in 1991, a European Silver medal in 1993 and a World Silver medal in 2000.

 In 1999, at the European Curling Championships in Chamonix, as well as receiving a Bronze medal, she was awarded the Fair Play & Friendly Award by her fellow
athletes for her conduct and attitude, on and off the ice.
</item 6>

<item 7>
2010 Toray Pan Pacific Open – Doubles
Alisa Kleybanova and Francesca Schiavone were the defending champions.

 Both were present, but competed with different partners.Kleybanova competed with Hsieh Su-wei, but lost in the first round to Benešová and Záhlavová-Strýcová,
while Schiavone competed with Tathiana Garbin, however withdrew before their quarterfinal match against Lisa Raymond and Rennae Stubbs.
</item 7>

<item 8>
2010 US Open – Mixed Doubles
Carly Gullickson and Travis Parrott were the defending champions, but lost in the first round to Gisela Dulko and Pablo Cuevas 2–6, 4–6.
</item 8>

<item 9>
2006 Qatar Ladies Open – Doubles
Alicia Molik and Francesca Schiavone were the defending champions, but Molik did not compete this year.

 Schiavone teamed up with Květa Peschke and lost in querterfinals to Elena Likhovtseva and Vera Zvonareva.
</item 9>

<item 10>
The tennis reports mainly track defending doubles champions who withdrew, split partners, or exited early. Chan Yung-jan and Abigail Spears won the 2009 Korea
Open; Sania Mirza and Mahesh Bhupathi withdrew from the 2010 Australian Open mixed doubles, while defending US Open champions Carly Gullickson and Travis
Parrott lost in round one. Francesca Schiavone won the 2010 French Open and reached world No. 4. Lee Chong Wei and
</item 10>
=== END ITEM id=c66c36f739a8fa6e ===

=== ITEM id=e2e734df96498ed7 ===
Gold sentences and facts:
Sentence 0: Mauritz Stiller (17 July 1883 – 18 November 1928) was a Finnish-Swedish film director, best known for discovering Greta Garbo and bringing her to America.
  Fact 0.0: Mauritz Stiller was born on 17 July 1883.
  Fact 0.1: Mauritz Stiller died on 18 November 1928.
  Fact 0.2: Mauritz Stiller was a Finnish-Swedish film director.
  Fact 0.3: Mauritz Stiller was best known for discovering Greta Garbo.
  Fact 0.4: Mauritz Stiller brought Greta Garbo to America.
Sentence 1: Johannes Grenzfurthner (] ; born 1975 in Vienna) is an award-winning Austrian artist, filmmaker, writer, actor, curator, theatre director and lecturer.
  Fact 1.0: Johannes Grenzfurthner was born in 1975 in Vienna.
  Fact 1.1: Johannes Grenzfurthner is an award-winning Austrian artist.
  Fact 1.2: Johannes Grenzfurthner is an Austrian filmmaker.
  Fact 1.3: Johannes Grenzfurthner is an Austrian writer.
  Fact 1.4: Johannes Grenzfurthner is an Austrian actor.
  Fact 1.5: Johannes Grenzfurthner is an Austrian curator.
  Fact 1.6: Johannes Grenzfurthner is an Austrian theatre director.
  Fact 1.7: Johannes Grenzfurthner is an Austrian lecturer.

Retrieved items:
<item 1>
Mauritz Stiller
Mauritz Stiller (17 July 1883 – 18 November 1928) was a Finnish-Swedish film director, best known for discovering Greta Garbo and bringing her to America.
</item 1>

<item 2>
Graph query row:
person: Mauritz Stiller
nationality: Finnish-Swedish
</item 2>

<item 3>
Willi Forst
Willi Forst, born Wilhelm Anton Frohs (7 April 1903 – 11 August 1980) was an Austrian actor, screenwriter, film director, film producer and singer.

 As a debonair actor he was a darling of the German-speaking film audiences, as a director, one of the most significant makers of the Viennese period musical
melodramas and comedies of the 1930s known as "Wiener Filme".

 From the mid-1930s he also recorded many records, largely of sentimental Viennese songs, for the Odeon Records label owned by Carl Lindström AG.
</item 3>

<item 4>
Devid Striesow
Devid Striesow (born 1 October 1973 in Bergen auf Rügen) is a German actor.

 He starred as "Sturmbannführer Herzog" (Bernhard Krüger) in Stefan Ruzowitzky's 2007 film "The Counterfeiters", which was awarded the Academy Award for Best
Foreign Language Film for that year.
</item 4>

<item 5>
Stefan Ruzowitzky
Stefan Ruzowitzky (born 25 December 1961) is an Austrian film director and screenwriter.
</item 5>

<item 6>
Richard Genée
Franz Friedrich Richard Genée (February 7, 1823 – June 15, 1895) was a Prussian born Austrian librettist, playwright, and composer.

 He is most famous for the libretto of "Die Fledermaus", Johann Strauss II's most famous operetta.

 He co-wrote the libretto without having met top-billed librettist Karl Haffner, who constructed the new story based on a play by Henri Meilhac and Ludovic
Halévy, which was considered too shocking to perform outside Paris.

 Genée, however, wrote the operetta's actual text and drew nothing from Haffner beyond the names of the characters.
</item 6>

<item 7>
Stefan Walz
Stefan Walz (born 1963) is a Swiss actor.

 His best known role is as Sigmund Jähn in the film "Good Bye, Lenin!

".
</item 7>

<item 8>
Graph query row:
person: Johannes Grenzfurthner
nationality: Austrian
</item 8>

<item 9>
Traceroute (film)
Traceroute is a 2016 Austrian/American documentary film directed by Johannes Grenzfurthner.

 The autobiographical documentary and road movie deals with the history, politics and impact of nerd culture.

 Grenzfurthner calls his film a "personal journey into the uncharted depths of nerd culture, a realm full of dangers, creatures and more or less precarious
working conditions", an attempt to "chase the ghosts of nerddom's past, present and future."

 The film was co-produced by art group monochrom and Reisenbauer Film.

 It features music by Kasson Crooker, Hans Nieswandt, and many others.
</item 9>

<item 10>
Johannes Grenzfurthner
Johannes Grenzfurthner (] ; born 1975 in Vienna) is an award-winning Austrian artist, filmmaker, writer, actor, curator, theatre director and lecturer.

 He is known as the founder, conceptualist and artistic director of monochrom, an international art and theory group.

 Most of his artworks are labelled "monochrom".
</item 10>
=== END ITEM id=e2e734df96498ed7 ===

=== ITEM id=376b964c318c6cda ===
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
Robert Holdstock
Robert Paul Holdstock (2 August 1948 – 29 November 2009) was an English novelist and author best known for his works of Celtic, Nordic, Gothic and Pictish
fantasy literature, predominantly in the fantasy subgenre of mythic fiction.
</item 1>

<item 2>
Christopher Guest
Christopher Haden-Guest, 5th Baron Haden-Guest (born February 5, 1948), simply known as Christopher Guest, is an English-American screenwriter, composer,
musician, director, actor, and comedian who holds dual British and American citizenship.

 Guest is most widely known in Hollywood for having written, directed and starred in his series of comedy films shot in mock-documentary (mockumentary) style.

 Many scenes and character backgrounds in Guest's films are written and directed, although actors have no rehearsal time and the ensemble improvise scenes while
filming them.

 The series of films began with "This Is Spinal Tap" (directed by Rob Reiner), and continued with "Waiting for Guffman", "Best In Show", "A Mighty Wind", "For
Your Consideration", and "Mascots".
</item 2>

<item 3>
Robert Holdstock bibliography
This is a bibliography of fantasy author Robert Holdstock.
</item 3>

<item 4>
John Updike
John Hoyer Updike (March 18, 1932 – January 27, 2009) was an American novelist, poet, short story writer, art critic, and literary critic.

 One of only three writers to win the Pulitzer Prize for Fiction more than once (the others were Booth Tarkington and William Faulkner), Updike published more
than twenty novels, more than a dozen short-story collections, as well as poetry, art and literary criticism and children's books during his career.
</item 4>

<item 5>
Stanley Bing
Stanley Bing is the pen name of Gil Schwartz (born May 20, 1951 in New York, NY), a business humorist and novelist.

 He has written a column for "Fortune" magazine for more than twenty years, after having spent a decade at "Esquire".

 He is the author of thirteen books including "What Would Machiavelli Do?"

 and "The Curriculum", a satirical textbook for a business school that also offers lessons on the Web.

 Schwartz is the senior executive vice president of corporate communications and Chief Communications Officer for CBS.
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
The American Mercury
The American Mercury was an American magazine published from 1924 to 1981.

 It was founded as the brainchild of H. L. Mencken and drama critic George Jean Nathan.

 The magazine featured writing by some of the most important writers in the United States through the 1920s and 1930s.

 After a change in ownership in the 1940s, the magazine attracted conservative writers.

 A second change in ownership a decade later turned the magazine into a virulently anti-Semitic publication.

 It was published monthly in New York City.

 The magazine went out of business in 1981, having spent the last 25 years of its existence in decline and controversy.
</item 7>

<item 8>
Oswalt Kolle
Oswalt Kolle (2 October 1928, Kiel – 24 September 2010, Amsterdam) was a German sex educator, who became famous during the late 1960s and early 1970s for his
numerous pioneering books and films on human sexuality.

 His work was translated into all major languages, while his films found an audience of 140 million worldwide.

 In his 1997 book "Open to Both Sides" he came out as bisexual.

 He was awarded the Magnus Hirschfeld Medal in 2000.
</item 8>

<item 9>
H. L. Mencken
Henry Louis Mencken (September 12, 1880 – January 29, 1956) was an American journalist, satirist, cultural critic and scholar of American English.

 Known as the "Sage of Baltimore", he is regarded as one of the most influential American writers and prose stylists of the first half of the twentieth century.

 He commented widely on the social scene, literature, music, prominent politicians and contemporary movements.

 His satirical reporting on the Scopes trial, which he dubbed the "Monkey Trial", also gained him attention.
</item 9>

<item 10>
Adam Fox
Adam Fox (1883 – 1977), Canon, was the Dean of Divinity at Magdalen College, Oxford.

 He was one of the first members of the "Inklings", a literary group which also included C.S. Lewis and J.R.R. Tolkien.

 Between 1938 and 1942 he was Professor of Poetry.

 Later he became Canon of Westminster Abbey and he is buried there in Poet's Corner.

 He was also Warden (Headmaster) of the famous Radley College.
</item 10>
=== END ITEM id=376b964c318c6cda ===
