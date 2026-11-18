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

=== ITEM id=e28d505b76f9ed12 ===
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
Diva (magazine)
DIVA is a leading lesbian magazine in the United Kingdom, published monthly.

 It was launched in March 1994 by Millivres Ltd, under the editorship of Frances Williams and is now published by Twin Media Group, which is owned by Silke
Bader and Linda Riley.

 Silke Bader is also the publisher of Curve magazine in the USA and LOTL in Australia, and Linda Riley runs the European Diversity Awards and is the former
publisher of g3 magazine in the UK.
</item 1>

<item 2>
Essence (magazine)
Essence is a monthly magazine for African American women between the ages of 18 and 49.

 It is the only magazine that focuses on reaching an audience of black women, revolves around the black woman experience, and has remained for a long period of
time.

 The magazine covers fashion, lifestyle and beauty, with an intimate girlfriend-to-girlfriend tone, and its slogan "Fierce, Fun, and Fabulous" suggests the
magazine's goal of empowering African-American women.

 The topics the magazine discusses range from celebrities, to fashion, to point-of-view pieces addressing current issues in the African-American community.

 A number of its readers engage closely and personally with the publication, and it claims to be the magazine "for and about Black women".
</item 2>

<item 3>
Cult MTL
Cult MTL is an English language arts, culture and news website and monthly print publication, based in Montreal, Quebec, Canada.

 Its first print edition appeared on 7 September 2012.

 It was created only a few months after Montreal's last English-language alternative weekly, "Montreal Mirror", was unceremoniously closed by its parent
company, Quebecor.

 The founding editor's of "Cult MTL" were also involved with the "Mirror".

 In August 2013 the print version of the magazine was started.
</item 3>

<item 4>
La Belle Assemblée
La Belle Assemblée (in full La Belle Assemblée or, Bell's Court and Fashionable Magazine Addressed Particularly to the Ladies) was a British women's magazine
published from 1806 to 1837, founded by John Bell (1745–1831).
</item 4>

<item 5>
Hänt Bild (Swedish weekly, 2003–, Aller); Hänt Extra (Swedish celebrity/youth weekly); Girlfriends (lesbian culture, 1993–2006); Rap-Up (hip-hop/R&B quarterly,
2001–); Sculptural Pursuit (quarterly art); DIVA (UK lesbian monthly, 1994–); Elle (global fashion/lifestyle, founded 1945); Girls’ Life (US teen, 1994–
</item 5>

<item 6>
SexIs
SexIs Magazine is a quarterly print publication and daily webzine devoted to sex and sexual culture, founded in 2008 by Web Merchants, parent company of sex toy
e-tailer EdenFantasys.com.

 The first print issue debuted in November 2009, distributed nationwide as an insert in "BUST magazine".

 The website publishes articles, columns, video presentations and news items daily.
</item 6>

<item 7>
VIVmag
VIVmag is one of the earliest exclusively digital, interactive women's lifestyle magazines.

 Launched in 2006 and backed by Canadian industrialist David Harrison Gilmour (the founder of FIJI Water LLC), the bimonthly magazine is distributed by Zinio
Systems, Inc.

 The first issue appeared in January-February 2007.

 There is no print version.

 The paperless title can be viewed in most browsers or downloaded to your computer and read via the Zinio Reader.

 The magazine is based in Thousand Oaks, California.
</item 7>

<item 8>
Annales de chimie et de physique
Annales de chimie et de physique (French for "Annals of Chemistry and of Physics") is a scientific journal that was founded in Paris, France, in 1789 under the
title "Annales de chimie".

 One of the early editors was the French chemist Antoine Lavoisier.

 In 1815, it became the "Annales de chimie et de physique", and was published under that name for the next 100 years.
</item 8>

<item 9>
Elle (magazine)
Elle is a worldwide lifestyle magazine of French origin that focuses on fashion, beauty, health, and entertainment.

 "Elle" is also the world's best-selling fashion magazine.

 It was founded by Pierre Lazareff and his wife Hélène Gordon in 1945.

 The title, in French, means "she" or "her".
</item 9>

<item 10>
Auguste Brancart
August or Auguste Brancart (21 July 1851 - 1894?)

 was a Belgian publisher of pornographic literature, credited with the first publication of "My Secret Life".

 He published translations of English pornography into French and vice versa for English publishers such as Edward Avery.

 He also published work of the Decadent movement such as "Monsieur Vénus" by Rachilde.
</item 10>
=== END ITEM id=e28d505b76f9ed12 ===

=== ITEM id=2bf88a7ee24bfffb ===
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
Mirrors (film)
Mirrors is a 2008 supernatural horror film directed by Alexandre Aja, starring Kiefer Sutherland, Paula Patton, and Amy Smart.

 The film was first titled "Into the Mirror", but the name was later changed to "Mirrors".

 Filming began on May 1, 2007, and it was released in American theaters on August 15, 2008.
</item 1>

<item 2>
Illusions &amp; Mirrors
Illusions & Mirrors is a 2013 short film directed by Iranian-born American artist and filmmaker Shirin Neshat.

 The film, shot in black-and-white, stars Tarek Aylouch, Michael Markiewicz, and the Israeli-born American actress Natalie Portman.

 The film was commissioned by Dior, for which Portman is a spokesmodel, for the Miss Dior Expo at the Grand Palais, Paris.

 Shirin Neshat said that the film is a tribute to the black-and-white silent films made by such surrealist filmmakers as Man Ray, Jean Cocteau, Luis Buñuel, and
Maya Deren.
</item 2>

<item 3>
Mira Nair
Mira Nair (born 15 October 1957) is an Indian American filmmaker based in New York City.

 Her production company, Mirabai Films, specializes in films for international audiences on Indian society, whether in the economic, social or cultural spheres.

 Among her best known films are "Mississippi Masala", "The Namesake", the Golden Lion-winning "Monsoon Wedding" and "Salaam Bombay!

", which was nominated for the Academy Award for Best Foreign Language Film.
</item 3>

<item 4>
These Old Broads
These Old Broads is a 2001 television film written by Carrie Fisher and starring her mother Debbie Reynolds, as well as Shirley MacLaine, Joan Collins, and
Elizabeth Taylor in her final film role.

 In a 2001 BBC "Omnibus" documentary about Elizabeth Taylor, Shirley MacLaine says that Julie Andrews and Lauren Bacall were originally planned to be in the
movie.

 The role of Miriam Hodges was originally offered to June Allyson.
</item 4>

<item 5>
Jack &amp; Diane (film)
Jack & Diane is a 2012 American romantic horror film written and directed by Bradley Rust Gray, and starring Riley Keough and Juno Temple.

 Olivia Thirlby and Ellen Page were originally cast for the lead roles in 2008 before the project was postponed.
</item 5>

<item 6>
Ela, Hela and The Hitch
Ela, Hela, and the Hitch is a play by Václav Havel.

 The play was written for the Artistic Director of the Theatre on the Balustrade, Ivan Vyskočil, as part of a longer evening, entitled "Hitchhiking".

 Along with "Ela, Hela, and the Hitch", Havel also wrote a sketch called "Motormorphosis".

 Reportedly, Vyskočil altered Havel’s sketches for the performance, though the original text was discovered by a Czech theater scholar, Lenka Jungmannová.

 "Motormorphosis", in a translation by Carol Rocamora, was performed at the Havel Festival in 2006, a world premiere of the text as written.

 "Ela, Hela, and the Hitch" premiered in an English translation by Edward Einhorn following a revival of "Motormorphosis" at New York’s Bohemian National Hall
in 2011.
</item 6>

<item 7>
Zooey Deschanel filmography
American actress Zooey Deschanel made her film debut in the 1999 comedy feature "Mumford".

 She went on to gained public attention by co-starred in the comedy-drama "Almost Famous" (2000), the independent drama "Manic" (2001), opposite Joseph
Gordon-Levitt, and the comedy-drama "The Good Girl" (2002).

 She landed her first major role as a 18-year-old virgin in the romantic drama "All the Real Girls" (2003), for which she was nominated for an Independent
Spirit Award for Best Female Lead.

 Deschanel's biggest commercial success, as of 2017, came with the Christmas-fantasy film "Elf" (also in 2003), which grossed over $220 million worldwide.
</item 7>

<item 8>
Paula Patton
Paula Maxine Patton (born December 5, 1975) is an American actress.

 Patton made her film debut in the 2005 comedy film "Hitch" and starred in the epic fantasy film "Warcraft" (2016), based on the game series of the same name.

 Patton has also been the female lead in "Déjà Vu" (2006), "" (2011) and "2 Guns" (2013), and appeared in the critically acclaimed "Precious" (2009).
</item 8>

<item 9>
Hitch (film)
Hitch is a 2005 American romantic comedy film directed by Andy Tennant and starring Will Smith.

 The film, which was written by Kevin Bisch, co-stars Eva Mendes, Kevin James, and Amber Valletta.

 Smith plays the main fictional character of the film, Alex "Hitch" Hitchens, who is a professional dating consultant who makes a living teaching men how to woo
women.

 The film was released on February 11, 2005 by Columbia Pictures.
</item 9>

<item 10>
License to Wed
License to Wed is a 2007 American romantic comedy film starring Robin Williams, Mandy Moore and John Krasinski, and directed by Ken Kwapis.

 The film was released in theaters on July 3, 2007.
</item 10>
=== END ITEM id=2bf88a7ee24bfffb ===

=== ITEM id=639a6cd0bcf273cd ===
Gold sentences and facts:
Sentence 0: Pterostyrax, the epaulette tree, is a small genus of four species of deciduous large shrubs or small trees in the family Styracaceae, native to eastern Asia in China and Japan.
  Fact 0.0: Pterostyrax is also called the epaulette tree.
  Fact 0.1: Pterostyrax is a small genus.
  Fact 0.2: Pterostyrax has four species.
  Fact 0.3: Pterostyrax species are deciduous.
  Fact 0.4: Pterostyrax species are large shrubs or small trees.
  Fact 0.5: Pterostyrax is in the family Styracaceae.
  Fact 0.6: Pterostyrax is native to eastern Asia.
  Fact 0.7: Pterostyrax is native to China.
  Fact 0.8: Pterostyrax is native to Japan.
Sentence 1: Dregea is a genus of vines in the Apocynaceae, first described as a genus with this name in 1838.
  Fact 1.0: Dregea is a genus of vines.
  Fact 1.1: Dregea is in the family Apocynaceae.
  Fact 1.2: Dregea was first described as a genus with this name in 1838.
Sentence 2:  It is native to Africa and southern Asia.
  Fact 2.0: Dregea is native to Africa.
  Fact 2.1: Dregea is native to southern Asia.

Retrieved items:
<item 1>
Styrax litseoides
Styrax litseoides is a species of flowering plant in the genus "Styrax" and family Styracaceae.

 It is endemic to Vietnam.
</item 1>

<item 2>
Deutzia
Deutzia ( or ) is a genus of about 60 species of flowering plants in the family Hydrangeaceae, native to eastern and central Asia (from the Himalayas east to
Japan and the Philippines), and Central America and also Europe.

 By far the highest species diversity is in China, where 50 species occur.
</item 2>

<item 3>
Pterostyrax
Pterostyrax, the epaulette tree, is a small genus of four species of deciduous large shrubs or small trees in the family Styracaceae, native to eastern Asia in
China and Japan.

 They grow 4 - tall, with alternate, simple ovate leaves 6 - long and 4 - broad.

 The flowers are white, produced in dense panicles 8 - long.

 The fruit is an oblong dry drupe, with longitudinal ribs or narrow wings (the wings are absent in the related genus "Styrax", whence the name "Pterostyrax",
"winged styrax").
</item 3>

<item 4>
Pterostyrax hispidus
Pterostyrax hispidus, the epaulette tree or fragrant epaulette tree, is a species of flowering plant in the family Styracaceae, native to China and Japan.

 Growing to 15 m tall by 12 m broad, it is a substantial, spreading, deciduous shrub with oval leaves up to 20 cm long, and clusters of pure white, fragrant,
pendent flowers in summer.
</item 4>

<item 5>
Rhabdodontidae
Ősi "et al." (2012) proposed that "Rhabdodon" underwent gigantism on the mainland, as opposed to "Zalmoxes" and "Mochlodon" experiencing nanism on island
habitats.

 In 2013, Darren Naish discovered a tiny rhabdodont tibia from Sebeș, Romania, turning out to be the smallest known; a paper (Brusette et "al" 2013.)

 later in the same year identified it as "Zalmoxes", cf. "Z. shqiperorum".
</item 5>

<item 6>
Dregea
Dregea is a genus of vines in the Apocynaceae, first described as a genus with this name in 1838.

 It is native to Africa and southern Asia.
</item 6>

<item 7>
Dregea sinensis
Dregea sinensis is a species of plant in the Apocynaceae family that is native to China.
</item 7>

<item 8>
Rhabdodontidae
Rhabdodontids were herbivorous ornithopod dinosaurs from the Cretaceous Period.

 Rhabdodontids were similar to large, robust "hypsilophodonts", with deep skulls and jaws.

 The family was first proposed by David B. Weishampel and colleagues in 2002.

 The rhabdodontids were defined as "the most recent common ancestor of "Zalmoxes robustus" and "Rhabdodon priscus" and all the descendants of this common
ancestor".

 In 2005, Paul Sereno defined the family as "the most inclusive clade containing "Rhabdodon priscus" but not "Parasaurolophus walkeri"".

 The Rhabdodontidae includes the type genus "Rhabdodon", "Zalmoxes", "Mochlodon" and possibly "Muttaburrasaurus".

 Rhabdodontid fossils have been found in Europe and Australia in formations dating from the Early to Late Cretaceous, 100 to 66 million years ago.

 Ősi "et al."
</item 8>

<item 9>
Dregea volubilis
Dregea volubilis is a species of plant in the Apocynaceae family that is native to India.
</item 9>

<item 10>
Styrax crotonoides
Styrax crotonoides is a species of plant in the genus "Styrax" and family Styracaceae.

 It is found in Peninsular Malaysia and Singapore.
</item 10>
=== END ITEM id=639a6cd0bcf273cd ===

=== ITEM id=99e9302b348c9731 ===
Gold sentences and facts:
Sentence 0: Tritonia (flame freesia) is a genus of flowering plants in the iris family first described as a genus in 1802.
  Fact 0.0: Tritonia, also called flame freesia, is a genus of flowering plants.
  Fact 0.1: Tritonia belongs to the iris family.
  Fact 0.2: Tritonia was first described as a genus in 1802.
Sentence 1: Phyteuma is a genus of flowering plants in the family Campanulaceae, native to Europe and Morocco.
  Fact 1.0: Phyteuma is a genus of flowering plants.
  Fact 1.1: Phyteuma is in the family Campanulaceae.
  Fact 1.2: Phyteuma is native to Europe.
  Fact 1.3: Phyteuma is native to Morocco.

Retrieved items:
<item 1>
Tritonia (plant)
Tritonia (flame freesia) is a genus of flowering plants in the iris family first described as a genus in 1802.

 They are naturally distributed across southern Africa, with a high concentration of species in Cape Province of western South Africa.

 The genus is closely related to the genus "Ixia".
</item 1>

<item 2>
Tritoniopsis
Tritoniopsis is a genus of flowering plants in the family Iridaceae, first described as a genus in 1926.

 The entire genus is endemic to Cape Province in South Africa.

 The genus name refers to the African genus "Tritonia" and is combined with the Greek word "opsis", meaning "look-alike".
</item 2>

<item 3>
Phyteuma
Phyteuma is a genus of flowering plants in the family Campanulaceae, native to Europe and Morocco.
</item 3>

<item 4>
Zygotritonia
Zygotritonia is a genus of herbaceous, perennial and bulbous plants in the Iris family (Iridaceae).

 It contains four species distributed in Tropical Africa.

 The genus name is derived from the word zygomorphic (meaning bilabiate), and the apparent resemblance to some species in the genus "Tritonia".
</item 4>

<item 5>
Conophytum
Conophytum is a genus of South African and Namibian succulent plants that belong to the Aizoaceae family.

 The name is derived from the Latin “conus” (cone) and Greek “phytum” (plant).

 The plants are also known as knopies (buttons), waterblasies (water blisters), sphaeroids, conos, cone plants, dumplings, or button plants.
</item 5>

<item 6>
Thonningia
Thonningia is a monotypic genus of flowering plants in the family Balanophoraceae containing the single species Thonningia sanguinea.

 It is distributed throughout much of southern and western Africa, particularly the tropical regions.

 Common names for the plant include ground pineapple.

 A familiar plant to humans, it has an extremely long list of common names in many African languages.

 Many names are inspired by the resemblance of the plant's inflorescence to a pineapple or palm tree.

 Some of the names can be translated as pineapple of the bush (from Anyi), duiker's kolanut (from Igala), and crown of the ground (from Yoruba).
</item 6>

<item 7>
Caryophyllales
Caryophyllales ( ) is an order of flowering plants that includes the cacti, carnations, amaranths, ice plants, beets, and many carnivorous plants.

 Many members are succulent, having fleshy stems or leaves.
</item 7>

<item 8>
Vascular plant
Vascular plants (from Latin "vasculum": duct), also known as tracheophytes (from the equivalent Greek term "trachea") and also higher plants, form a large group
of plants (c. 308,312 accepted known species ) that are defined as those land plants that have lignified tissues (the xylem) for conducting water and minerals
throughout the plant.

 They also have a specialized non-lignified tissue (the phloem) to conduct products of photosynthesis.

 Vascular plants include the clubmosses, horsetails, ferns, gymnosperms (including conifers) and angiosperms (flowering plants).

 Scientific names for the group include Tracheophyta and Tracheobionta.
</item 8>

<item 9>
Ericaceae
The Ericaceae are a family of flowering plants, commonly known as the heath or heather family, found most commonly in acid and infertile growing conditions.

 The family is large, with c. 4250 known species spread across 124 genera, making it the 14th most species-rich family of flowering plants.

 The many well-known and economically important members of the Ericaceae include the cranberry, blueberry, huckleberry, rhododendron (including azaleas), and
various common heaths and heathers ("Erica", "Cassiope", "Daboecia", and "Calluna" for example).
</item 9>

<item 10>
Alismataceae
The water-plantains (Alismataceae) are a family of flowering plants, comprising 11 genera and between 85 and 95 species.

 The family has a cosmopolitan distribution, with the greatest number of species in temperate regions of the Northern Hemisphere.

 Most of the species are herbaceous aquatic plants growing in marshes and ponds.
</item 10>
=== END ITEM id=99e9302b348c9731 ===

=== ITEM id=dceebfda0063322c ===
Gold sentences and facts:
Sentence 0: George Albert Smith Sr. (April 4, 1870 – April 4, 1951) was an American religious leader who served as the eighth president of The Church of Jesus Christ of Latter-day Saints (LDS Church).
  Fact 0.0: George Albert Smith Sr. was born on April 4, 1870.
  Fact 0.1: George Albert Smith Sr. died on April 4, 1951.
  Fact 0.2: George Albert Smith Sr. was an American religious leader.
  Fact 0.3: George Albert Smith Sr. served as the eighth president of The Church of Jesus Christ of Latter-day Saints (LDS Church).
Sentence 1: Wong Jing ( born 3 May 1955) is a Hong Kong film director, producer, actor, presenter, and screenwriter.
  Fact 1.0: Wong Jing was born on 3 May 1955.
  Fact 1.1: Wong Jing is a Hong Kong film director.
  Fact 1.2: Wong Jing is a Hong Kong film producer.
  Fact 1.3: Wong Jing is a Hong Kong actor.
  Fact 1.4: Wong Jing is a Hong Kong presenter.
  Fact 1.5: Wong Jing is a Hong Kong screenwriter.

Retrieved items:
<item 1>
Old Church of St George, Hill Street Dublin
The Old Church of St. George, commonly called "Little George's" in Hill Street (formerly Temple Street Lower) Parish of St. Mary, Dublin was built in 1668 by
the Eccles family for their workmen and also as a chapel-of-ease to the nearby St. Mary's Church.

 However, that St. Mary's Church was not St. Mary's Church, Dublin as its foundation stone was laid in 1700, and it was not St. Mary's Abbey, Dublin as it was
dissolved in 1539.

 Therefore St George's church, Hill Street, may have been a Chapel-of-Ease to St. Michan's Church in Church Street.

 The main body of the church, with the exception of the tower, was demolished in 1894.
</item 1>

<item 2>
Graph query row:
person: Wong Jing
occupation: ['film director', 'producer', 'actor', 'presenter', 'screenwriter']
relationship: None
connected_entity: None
role: None
period: None
status: None
</item 2>

<item 3>
Angelino Apelar
Angelino "Andy" Baldemor Apelar (June 16, 1927 - January 23, 2006) was an Evangelical Christian leader in the Filipino American community, and a founder of the
Association of Filipino Churches (AFC) of the Christian and Missionary Alliance.

 He had served as a missionary/pastor and radio broadcaster in the Philippines before moving to the United States to start the AFC.

 He served as president for several terms.

 He graduated from the Far Eastern Bible Institute and Seminary, where he also met his future wife, Purita Palomar, a member of one of Capiz province's
influential families.

 They have two children: Faith and Jemuel Apelar.
</item 3>

<item 4>
Albert Smith Medal
The Albert Smith Memorial Medal is the award given to the Man of the Match in the final of the Camanachd Cup, the blue riband trophy of the sport of shinty.

 It has been presented every year since 1972 by the Smith family of Fort William in honour of Albert Smith Sr. (Born in 1888, Lochuanagan, Fort Augustus.)

 The widow of his only son, Albert Smith Jr. currently presents the medal.

 Albert Sr.'s grandson, Victor Smith is a former player for Fort William Shinty Club but never won the medal despite featuring as a key player in several Fort
William wins.
</item 4>

<item 5>
George Albert Smith
George Albert Smith Sr. (April 4, 1870 – April 4, 1951) was an American religious leader who served as the eighth president of The Church of Jesus Christ of
Latter-day Saints (LDS Church).
</item 5>

<item 6>
D. Arthur Haycock
David Arthur Haycock (September 4, 1916 – February 25, 1994) was a personal secretary to several twentieth-century presidents of The Church of Jesus Christ of
Latter-day Saints (LDS Church), including George Albert Smith, Joseph Fielding Smith, Harold B. Lee, Spencer W. Kimball and Ezra Taft Benson.
</item 6>

<item 7>
Graph query row:
person: George Albert Smith Sr.
occupation: religious leader
relationship: MEMBER_OF
connected_entity: The Church of Jesus Christ of Latter-day Saints
role: eighth president
period: None
status: None
</item 7>

<item 8>
Wong Jing
Wong Jing ( born 3 May 1955) is a Hong Kong film director, producer, actor, presenter, and screenwriter.

 A prolific filmmaker with strong instincts for crowd-pleasing and publicity, Wong Jin has played a prominent role in the Hong Kong cinema of the last
quarter-century.
</item 8>

<item 9>
George A. Smith
George Albert Smith (June 26, 1817 – September 1, 1875) (known throughout his life as George A. Smith) was an early leader in the Latter-day Saint movement.

 He served in the Quorum of the Twelve Apostles and as a member of the First Presidency of The Church of Jesus Christ of Latter-day Saints (LDS Church).
</item 9>

<item 10>
Smith Fieldhouse
The George Albert Smith Fieldhouse is a 5,000 seat multi-purpose arena in Provo, Utah.

 Built in 1951, it is the home of the Brigham Young University Cougars volleyball teams and most home gymnastics meets.

 It was named for George Albert Smith, the eighth president of The Church of Jesus Christ of Latter-day Saints, who died the year the fieldhouse opened.

 Prior to the Marriott Center opening in 1971 it was home to the basketball teams.

 At that time, the arena held 10,500 people.

 Smith Fieldhouse also has a track and several offices used by BYU's athletic department.
</item 10>
=== END ITEM id=dceebfda0063322c ===

=== ITEM id=87a60681f2e087ab ===
Gold sentences and facts:
Sentence 0: Ferdi Taygan (born December 5, 1956), is a former professional tennis player from the United States.
  Fact 0.0: Ferdi Taygan was born on December 5, 1956
  Fact 0.1: Ferdi Taygan is a former professional tennis player
  Fact 0.2: Ferdi Taygan is from the United States
Sentence 1: Mahesh Shrinivas Bhupathi (born 7 June 1974) is a retired Indian professional tennis player.
  Fact 1.0: Mahesh Shrinivas Bhupathi was born on 7 June 1974
  Fact 1.1: Mahesh Shrinivas Bhupathi is a retired Indian professional tennis player

Retrieved items:
<item 1>
1983 Volvo International – Doubles
Sherwood Stewart and Ferdi Taygan were the defending champions but they competed with different partners that year, Stewart with Mark Edmondson and Taygan with
Cássio Motta.
</item 1>

<item 2>
Mahesh Bhupathi
Mahesh Shrinivas Bhupathi (born 7 June 1974) is a retired Indian professional tennis player.

 In 1997, he became the first Indian to win a Grand Slam tournament (with Rika Hiraki).

 With his win at the Australian Open mixed doubles in 2006, he joined the elite group of eight tennis players who have achieved a career Grand Slam in mixed
doubles.

 He is also the founder of International Premier Tennis League.

 In December 2016, Bhupathi was appointed as India's next non-playing Davis Cup captain and took over the reins from Anand Amritraj in February 2017.
</item 2>

<item 3>
Ferdi Taygan
Ferdi Taygan (born December 5, 1956), is a former professional tennis player from the United States.

 He is of Turkish descent.
</item 3>

<item 4>
1983 Donnay Indoor Championships – Doubles
Pavel Složil and Sherwood Stewart were the defending champions, but Složil did not participate this year.

 Stewart partnered Ferdi Taygan, losing in the first round.
</item 4>

<item 5>
1982 Volvo Masters – Doubles
Peter Fleming and John McEnroe won in the final 6–4, 6–3 against Sherwood Stewart and Ferdi Taygan.
</item 5>

<item 6>
1982 French Open – Men's Doubles
The Men's Doubles tournament at the 1982 French Open was held from 24 May until 6 June 1982 on the outdoor clay courts at the Stade Roland Garros in Paris,
France.

 Sherwood Stewart and Ferdi Taygan won the title, defeating Hans Gildemeister and Belus Prajoux in the final.
</item 6>

<item 7>
1982 US Open – Mixed Doubles
Anne Smith and Kevin Curren were the defending champions and won in the final 6–7, 7–6 (7–4), 7–6 (7–5) against Barbara Potter and Ferdi Taygan.
</item 7>

<item 8>
1982 U.S. Pro Indoor – Doubles
Marty Riessen and Sherwood Stewart were the defending champions, but Riessen did not participate this year.

 Stewart partnered Ferdi Taygan, finishing runner-up.
</item 8>

<item 9>
Mahesh Bhupathi Tennis Academy
Mahesh Bhupathi Tennis Academy is a tennis training facility started by Indian tennis player Mahesh Bhupathi.

 The Academy has facilities across 9 states in India as well as in the UAE.
</item 9>

<item 10>
Gerald Melzer
Gerald Melzer (born 13 July 1990) is a professional Austrian tennis player.

 As a qualifier, he reached the semifinals of Münich in 2015.

 He achieved a career-high singles ranking of world No. 68 in November 2016.
</item 10>
=== END ITEM id=87a60681f2e087ab ===

=== ITEM id=8d5970d7eff469ae ===
Gold sentences and facts:
Sentence 0: Race to Nowhere is a 2009 documentary film written by Maimone Attia and directed by Vicki Abeles and Jessica Congdon.
  Fact 0.0: Race to Nowhere is a 2009 documentary film.
  Fact 0.1: Race to Nowhere was written by Maimone Attia.
  Fact 0.2: Race to Nowhere was directed by Vicki Abeles.
  Fact 0.3: Race to Nowhere was directed by Jessica Congdon.
Sentence 1: SlamNation is a documentary film by director Paul Devlin.
  Fact 1.0: SlamNation is a documentary film.
  Fact 1.1: SlamNation was directed by Paul Devlin.
Sentence 2:  The film follows the National Poetry Slam in Portland, Oregon.
  Fact 2.0: SlamNation follows the National Poetry Slam.
  Fact 2.1: The National Poetry Slam is in Portland, Oregon.

Retrieved items:
<item 1>
A Summer in the Cage
A Summer in the Cage is a 2007 documentary film about a man's experiences with bipolar disorder.

 The film follows the filmmaker's friend Sam and features an interview with mental health scholar Kay Redfield Jamison.

 It was directed by Benjamin Selkow.

 The documentary debuted on the Sundance Channel in 2007.
</item 1>

<item 2>
The End of America (film)
The End of America is a 2008 documentary film directed by Annie Sundberg and Ricki Stern, adapted from Naomi Wolf's 2007 book, "".

 The film analyzes policy changes made during the Bush Administration, and makes the case that these changes threaten American democracy.
</item 2>

<item 3>
We Are Poets
We Are Poets is a documentary film directed by Daniel Lucchesi and Alex Ramseyer-Bache.

 It follows six young poets from Leeds Young Authors performance poetry group in Leeds on their visit to the international Poetry Slam Competition in Washington
D.C, United States.

 It premièred at Europe's largest Documentary Film festival - Doc/Fest in Sheffield's Showroom Cinema on 11 June 2011, where it won the coveted Sheffield Youth
Jury Award.
</item 3>

<item 4>
Alice Walker: Beauty in Truth
Alice Walker: Beauty in Truth is a documentary film directed by Pratibha Parmar, made by Kali Films production company.

 The film follows the life of the Pulitzer Prize-winning author, poet and activist Alice Walker.

 Shooting began in May 2011.

 It was aired on the BBC on Monday July 8, 2013, and on PBS on February 7, 2014.
</item 4>

<item 5>
SlamNation
SlamNation is a documentary film by director Paul Devlin.

 The film follows the National Poetry Slam in Portland, Oregon.
</item 5>

<item 6>
Beyond This Place (2010 film)
Beyond This Place (2010) is a documentary film directed by Kaleo La Belle, screened at a number of film festivals.

 In it, La Belle reunites with his biological father Cloud Rock La Belle, a charismatic figure who was largely absent from his life.

 The film deals, in a personal and intimate way, with issues of parenting, issues of freedom versus responsibility, and with the aging of the 60's generation.
</item 6>

<item 7>
Punk's Not Dead (2007 film)
Punk's Not Dead is a 2007 documentary film directed by Susan Dynner, an American hardcore punk fan.

 The film claims to infiltrate American clubs, malls, recording studios, etc. where it sets out to claim hardcore punk and pop punk music is "thriving" from an
American perspective.

 Its content features performances largely from 1980s hardcore bands and MTV skate punk and pop punk/rock acts.

 It also includes various interviews and behind-the-scenes footage with the bands, labels and fans.
</item 7>

<item 8>
War/Dance
War/Dance is a 2007 American documentary film written and directed by Sean Fine and Andrea Nix Fine and produced by Shine Global's Susan MacLaury, a professor
at Kean University, and Albie Hecht.

 It was nominated for the 2008 Academy Award for Best Documentary Feature and received the Emmy Award for Best Documentary and Best Cinematography in 2010.
</item 8>

<item 9>
Poetry in Motion (film)
Poetry in Motion is a 1982 Canadian documentary film directed by Ron Mann featuring contemporary North American poetry and music.

 Featured are some of the Black Mountain poets, Beats, minimalist poets, and avant-garde poets.

 It was released in theaters, later being distributed on VHS, LaserDisc, and DVD.

 An extended CD-ROM version was also released.
</item 9>

<item 10>
Race to Nowhere
Race to Nowhere is a 2009 documentary film written by Maimone Attia and directed by Vicki Abeles and Jessica Congdon.
</item 10>
=== END ITEM id=8d5970d7eff469ae ===

=== ITEM id=2e910cf10a921cc4 ===
Gold sentences and facts:
Sentence 0:  His primary jockey is Tony McCoy.
  Fact 0.0: The unnamed subject of this sentence has Tony McCoy as its primary jockey.
  Fact 0.1: Tony McCoy is a primary jockey.
Sentence 1:  Based in Ireland and the UK, McCoy rode a record 4,358 winners, and was Champion Jockey a record 20 consecutive times, every year he was a professional.
  Fact 1.0: Tony McCoy is based in Ireland and the UK.
  Fact 1.1: Tony McCoy rode a record 4,358 winners.
  Fact 1.2: Tony McCoy was Champion Jockey a record 20 consecutive times.
  Fact 1.3: Tony McCoy was Champion Jockey every year he was a professional.

Retrieved items:
<item 1>
Fred Winter
Frederick Thomas Winter CBE (20 September 1926 – 5 April 2004) was a British National Hunt racing racehorse jockey and trainer.

 He was British jump racing Champion Jockey four times and British jump racing Champion Trainer eight times.

 He is the only person to have won the Cheltenham Gold Cup, Champion Hurdle and Grand National as both jockey and trainer.

 Winter won the Grand National four times, as a jockey in 1957 (Sundew) and 1962 (Kilmore), and as a trainer in 1965 (Jay Trump) and 1966 (Anglo).
</item 1>

<item 2>
Binocular (horse)
Binocular, foaled on March 17, 2004 in France, is a French thoroughbred retired racehorse, born to sire, Enrique and dam Bleu Ciel Et Blanc.

 He is owned by J. P. McManus and trained by Nicky Henderson.

 His primary jockey is Tony McCoy.
</item 2>

<item 3>
Ernest Piggott
Ernest "Ernie" Piggott (1878–1967) born Nantwich, Cheshire, England was a leading British jump racing jockey, whose family has become one of the leading
dynasties in British horseracing.

 He was three times Champion Jockey and three times Grand National winner.

 His son, (Ernest) Keith Piggott (1904–1993), was also a leading jump jockey and National-winning trainer, while his grandson is the 11-times British flat
racing Champion Jockey, Lester Piggott.
</item 3>

<item 4>
Irish flat racing Champion Jockey
The Champion Jockey of flat racing in Ireland is the jockey who has ridden the most winning horses during a season.

 The list below shows the Champion Jockey for each year since 1950.
</item 4>

<item 5>
Irish jump racing Champion Jockey
The Champion Jockey of National Hunt racing in Ireland is the jockey who has ridden the most winning horses during a season.

 The list below shows the Champion Jockey for each year since 1947.

 The current champion, as of 2017, is Ruby Walsh.
</item 5>

<item 6>
Bart Cummings
James Bartholomew Cummings {'1': ", '2': ", '3': ", '4': "} (14 November 1927 – 30 August 2015), also known by his initials J. B. Cummings, was one of the most
successful Australian racehorse trainers.

 He was known as the "Cups King", referring to the Melbourne Cup, as he won 'the race that stops a nation' a record twelve times.

 During his lifetime Cummings was considered an Australian cultural icon and an Australian National Living Treasure.
</item 6>

<item 7>
British flat racing Champion Jockey
The Champion Jockey of flat racing in Great Britain is the jockey who has ridden the most winning horses during a season.

 The list below shows the Champion Jockey and the number of winners for each year since 1840.

 The seasonal record of jockeys' winners was published for the first time in 1846.

 The championship was sponsored for the first time in 2009 by online casino 32Red and is currently sponsored by Stobart Group.

 The 2016 title was determined by the number of winners ridden between 30 April and 15 October, in both turf and all-weather racing.

 Prior to 2015 the title was decided over the full length of the Flat turf season, from March to November, and carried no prize money to the winner.

 A prize of £25,000 to the champion jockey was also introduced as part of the 2015 changes.
</item 7>

<item 8>
Gordon Richards
Sir Gordon Richards (5 May 1904 – 10 November 1986) was an English jockey.

 He was the British flat racing Champion Jockey 26 times and is often considered the world's greatest ever jockey.

 He remains the only flat jockey to have been knighted.
</item 8>

<item 9>
German flat racing Champion Jockey
The Champion Jockey of flat racing in Germany is the jockey who has ridden the most winning horses during a season.

 The list of the Champion Jockey started 1870.

 The list below shows the Champion Jockey for each year since 1984.
</item 9>

<item 10>
Tony McCoy
Sir Anthony Peter McCoy, OBE (born 4 May 1974), commonly known as A. P. McCoy or Tony McCoy, is a Northern Irish former horse racing jockey.

 Based in Ireland and the UK, McCoy rode a record 4,358 winners, and was Champion Jockey a record 20 consecutive times, every year he was a professional.

 He stands 1.78 m (5'10"), far taller than most jockeys.
</item 10>
=== END ITEM id=2e910cf10a921cc4 ===

=== ITEM id=4fbc28c4cbdf072c ===
Gold sentences and facts:
Sentence 0: The Australian women's national soccer team represents Australia in international women's soccer at the senior level.
  Fact 0.0: The Australian women's national soccer team represents Australia in international women's soccer.
  Fact 0.1: The Australian women's national soccer team represents Australia at the senior level.
Sentence 1:  The team's official nickname is the Matildas (from the Australian folk song "Waltzing Matilda"), having been known as the Female Socceroos before 1995.
  Fact 1.0: The Australian women's national soccer team's official nickname is the Matildas.
  Fact 1.1: The nickname Matildas comes from the Australian folk song "Waltzing Matilda".
  Fact 1.2: The Australian women's national soccer team was known as the Female Socceroos before 1995.
Sentence 2: "Waltzing Matilda" is Australia's best-known bush ballad, and has been described as the country's "unofficial national anthem".
  Fact 2.0: "Waltzing Matilda" is Australia's best-known bush ballad.
  Fact 2.1: "Waltzing Matilda" has been described as the country's "unofficial national anthem."

Retrieved items:
<item 1>
Spencer Park, Queensland
Spencer Park (currently known as Corporate Travel Management Stadium) is a football stadium located in the suburb of Newmarket, north of Brisbane, Queensland,
Australia.

 It is home to National Premier Leagues Queensland team Brisbane City FC.

 The stadium, while officially holding 10,000, seats only 3,000 under the grandstand, the "Gino Merlo Stand".

 The stadium opened in 1963 on the site of an old Brisbane City Council waste disposal site, and underwent a major renovation in 1981 with the construction of
the Gino Merlo Stand.

 The stadium is home to La Rustica restaurant, where traditional Italian foods, including pizza, are served.
</item 1>

<item 2>
RPA &amp; The United Nations of Sound
The band played in Australia also during the Splendour in the Grass Festival in late July.

 The band's debut album as Richard's backing lineup, entitled "United Nations of Sound", was released on 19 July 2010.

 "Born Again", the first proper single, was officially released the same day of the album as a digital download.
</item 2>

<item 3>
Australia at the 2015 Pacific Games
Australia competed at the 2015 Pacific Games in Port Moresby, Papua New Guinea from 4 to 18 July 2015.

 Australia qualified 43 athletes.

 It was the first time that Australia has competed in the Pacific Games.
</item 3>

<item 4>
Waltzing Matilda
"Waltzing Matilda" is Australia's best-known bush ballad, and has been described as the country's "unofficial national anthem".
</item 4>

<item 5>
Anigozanthos manglesii
Anigozanthos manglesii, commonly known as the red-and-green kangaroo paw, Mangles kangaroo paw, Kurulbrang(Noongar) is a plant species endemic to Western
Australia, and the floral emblem of that state.
</item 5>

<item 6>
Anigozanthos flavidus
Anigozanthos flavidus is a species of plant found in Southwest Australia.

 It is member of the Haemodoraceae family.

 It is commonly known as the tall, yellow, or evergreen, kangaroo paw.

 The specific epithet, "flavidus", refers to the yellow flowers of this plant.
</item 6>

<item 7>
Vanuatu national under-23 football team
The Vanuatu national under-23 football team, also known as Vanuatu Cyclone, represents Vanuatu at U23 tournaments.

 The team is considered to be the feeder team for the Vanuatu national football team.

 They are controlled by the Vanuatu Football Federation.

 The team has gained notoriety for thrashing Micronesia's side 46–0 in the 2015 Pacific Games.
</item 7>

<item 8>
Australia women's national soccer team
The Australian women's national soccer team represents Australia in international women's soccer at the senior level.

 The team is overseen by the governing body for soccer in Australia, Football Federation Australia (FFA), which is currently a member of the Asian Football
Confederation (AFC) and the regional ASEAN Football Federation (AFF) since leaving the Oceania Football Confederation (OFC) in 2006.

 The team's official nickname is the Matildas (from the Australian folk song "Waltzing Matilda"), having been known as the Female Socceroos before 1995.

 The current team manager is Alen Stajcic.
</item 8>

<item 9>
Australia (Gyroscope song)
"Australia" is a song by Australian rock band Gyroscope, from the album "Breed Obsession".

 It was released as the third single from their highly successful album, which peaked at number one on the Australian Albums Chart.

 The song was released to radio in May 2008, and was released as a CD single and digital download on 14 June 2008, along with two acoustic pieces.

 One of those being a live acoustic version of "Australia", the other a live acoustic cover of Chris Isaak's "Wicked Game".
</item 9>

<item 10>
Wangaratta Wahine
Wangaratta Wahine is an album by Australian band The Captain Matchbox Whoopee Band, released in 1974 and was their second album.

 It was recorded at top recording studio Armstrong Studios in Melbourne and the cover art was by famed Australian artist Michael Leunig.
</item 10>
=== END ITEM id=4fbc28c4cbdf072c ===

=== ITEM id=81701c03c99685e1 ===
Gold sentences and facts:
Sentence 0: Black (Bengali: ব্ল্যাক ) is a Bangladeshi rock band, formed in Dhaka, Bangladesh.
  Fact 0.0: Black is a Bangladeshi rock band.
  Fact 0.1: Black was formed in Dhaka, Bangladesh.
  Fact 0.2: The Bengali name of Black is ব্ল্যাক.
Sentence 1: New Found Glory (formerly A New Found Glory) is an American rock band from Coral Springs, Florida, formed in 1997.
  Fact 1.0: New Found Glory is an American rock band.
  Fact 1.1: New Found Glory was formerly known as A New Found Glory.
  Fact 1.2: New Found Glory is from Coral Springs, Florida.
  Fact 1.3: New Found Glory was formed in 1997.

Retrieved items:
<item 1>
New Found Glory
New Found Glory (formerly A New Found Glory) is an American rock band from Coral Springs, Florida, formed in 1997.

 The band currently consists of Jordan Pundik (lead vocals), Ian Grushka (bass guitar), Chad Gilbert (lead guitar, backing vocals), and Cyrus Bolooki (drums).

 Longtime rhythm guitarist and lyricist Steve Klein departed from the band in late 2013, following "personal differences."

 During their lengthy recording career, the band have released nine studio albums, one live album, two EPs, and three cover albums.
</item 1>

<item 2>
It's All About the Girls
It's All About the Girls is the debut EP by A New Found Glory (now New Found Glory) released on December 20, 1997 by Fiddler Records.

 It is the only release with their original drummer Joe Marino.

 A re-release, featuring revised cover artwork, was issued by Fiddler in June, 2000.
</item 2>

<item 3>
Radiosurgery (album)
Radiosurgery is the seventh studio album by American rock band New Found Glory.

 It was first released on September 30, 2011 in Australia, before its wider release on October 4 through independent label Epitaph Records.

 It is the band's final studio album to feature founding guitarist Steve Klein.

 To follow up predecessor "Not Without a Fight" (2009), the band began writing new material during their stint on the 2010 Honda Civic Tour.

 After self-producing a set of demos and contacting long-term record producer Neal Avron, the band went on to record the album in Avron's home recording studio
over a period of three months in 2011.

 The quintet set out to write an album that paid homage to classic punk rock records that first inspired them to form a band during the 1990s.

 Listening extensively to the likes of early Green Day and Ramones, New Found Glory strove to create a sound that could "bridge the gap" between old and new
generations of the genre.
</item 3>

<item 4>
Clearview 77
Clearview 77 was a punk rock quartet hailing from Brooklyn, NY.

 Born in the summer of 1996, CV77 was the true definition of the "local" band.

 The band has played countless shows within the East Coast with acts like Face to Face, New Found Glory, Finch, Saves the Day, The Amazing Crowns, US Bombs, Big
Wig, Murphy's Law, Alkaline Trio, The Queers, GOB, and many more.

 In their 8-year existence the band managed to finish at the top of the charts and competitions for CMJ, Discmakers, Garageband.com, mp3.com, Jim Beam,
Loudenergy, had a song aired on CNN and Good Morning America, and added to college radio stations across the US.

 The band also finished in the top 10 in 92.3 FM K-ROCK' best unsigned band of 2000 contest.

 Clearview 77 have also been featured on multiple compilations with acts like NOFX, New Found Glory, Auto Pilot Off, Agnostic Front, Midtown, ALL, Saves the
Day, Down By Law, Circle Jerks, and many more.
</item 4>

<item 5>
New Found Glory, formed in Florida in 1997, released the self-produced, cassette-only *A Very New Found Glory Christmas* in 2012: two originals, three covers,
2,000 copies, with *Home Alone*-parody artwork. Founding guitarist Steve Klein left in 2013. Their releases include 1999 debut *Nothing Gold Can Stay*,
soundtrack-cover EP *From the Screen to Your Stereo* (2000), and *Radiosurgery* (201
</item 5>

<item 6>
From the Screen to Your Stereo
From the Screen to Your Stereo is an EP by A New Found Glory (now New Found Glory) released on 28 March 2000 by Drive-Thru Records.

 All the tracks are covers of songs from motion picture soundtracks.

 The album was pressed on 10" vinyl and released in three different colors: light marble blue, marble red, and white.

 The vinyl pressing featured a bonus track.

 In 2002, the album was released on a non-US split with the Japanese band Nicotine entitled Movie Addiction, where on this split, Nicotine also covered songs
from movies.

 In 2007 the band released a follow up, this time a full-length album titled "From the Screen to Your Stereo Part II".
</item 6>

<item 7>
A Very New Found Glory Christmas
A Very New Found Glory Christmas is the third EP by American rock band New Found Glory.

 Self-produced by the band, it is a special edition Christmas-themed acoustic recording featuring two original songs and three covers.

 The EP was limited to 2,000 pressings and was released via cassette tape on December 1, 2012.

 1,000 white copies were sold on tour, and 1,000 red copies were sold in their online store.

 The band released "Nothing for Christmas" via Rdio and Spotify on December 20, after debuting songs off the EP during the tenth anniversary tour of their
"Sticks and Stones" album.

 The cover art is a parody of the 1990 blockbuster movie "Home Alone".
</item 7>

<item 8>
Hit or Miss (New Found Glory song)
"Hit or Miss (Waited Too Long)" is the first single by New Found Glory (formerly A New Found Glory), from their 1999 debut studio album, "Nothing Gold Can
Stay".
</item 8>

<item 9>
Chad Gilbert
Chad Everett Gilbert (born March 9, 1981) is an American musician and record producer.

 He is a founding member of the rock band New Found Glory, for whom he plays lead guitar, composes music and provides backing vocals.

 He was also the lead vocalist for New Found Glory's now-defunct side-project, International Superheroes of Hardcore.

 Additionally, Gilbert was the vocalist for the hardcore punk band Shai Hulud between 1995 and 1998, and he rejoined the band in 2012.
</item 9>

<item 10>
Nothing Gold Can Stay (album)
Nothing Gold Can Stay is the debut studio album by American rock band New Found Glory, released on October 19, 1999 through independent record label Eulogy
Recordings.

 At the time, the band was then named "A New Found Glory", but later dropped the indefinite article "A" due to some fans struggling to find their records in
stores.

 The original pressings of the album contained samplings from several films including "The Outsiders" (1983), "Weird Science" (1985), and "That Thing You Do!

" (1996), as well as Robert Frost's poem "Nothing Gold Can Stay", after which the album is named.
</item 10>
=== END ITEM id=81701c03c99685e1 ===

=== ITEM id=d8ce2ae5d7e73271 ===
Gold sentences and facts:
Sentence 0:  In 1972, a grant from the Eleanor and John Y. Brown Jr., Foundation led to the construction of the current facility.
  Fact 0.0: The Eleanor and John Y. Brown Jr. Foundation gave a grant in 1972.
  Fact 0.1: The 1972 grant from the Eleanor and John Y. Brown Jr. Foundation led to the construction of the current facility.
Sentence 1:  He served as the 55th governor of Kentucky from 1979 to 1983, although he may be best known for building Kentucky Fried Chicken (KFC) into a multimillion-dollar restaurant chain.
  Fact 1.0: The 55th governor of Kentucky served from 1979 to 1983.
  Fact 1.1: The 55th governor of Kentucky may be best known for building Kentucky Fried Chicken (KFC) into a multimillion-dollar restaurant chain.

Retrieved items:
<item 1>
Rubin Schron
Other buildings he owns include the Monterey, a 521-unit rental multifamily building on Manhattan's Upper East Side; over the decades, Schron has also amassed a
portfolio of Mitchell-Lama apartment buildings whose values have been skyrocketing to record values, after reverting to market rates when government subsidies
expired.

 In 2007, he sold nearly 4,000 units of former Mitchell-Lama properties in five complexes in Harlem and on Roosevelt Island for $940 million.

 Schron, who practices Orthodox Judaism, has eight children and 50 grandchildren.

 He and his family have lived in the same single-family home in Brooklyn for many decades.
</item 1>

<item 2>
Rubin Schron
Rubin Schron, who goes by Ruby, is a New York City real estate investor, landlord and the founder of Cammeby's International Group.

 He has a personal net worth of well over $10 billion, according to data company Real Capital Analytics.

 The portfolio of Cammeby's, which Schron founded in 1967, includes office buildings, market-rate and government-subsidized apartment complexes, nursing homes,
the 16-building complex in Sunset Park now known as Industry City, a stake in the bottom half of Woolworth Building and industrial properties scattered across
Long Island.

 In 2013, Schron made an unsolicited and unsuccessful offer to buy the Empire State Building for $2 billion, but he has not had problems closing many other
deals throughout his long career.

 In 2003, an investment group led by Schron paid $600 million for a portfolio of about 6,000 outer-borough apartments from Donald Trump.
</item 2>

<item 3>
McAlister's Deli
McAlister's Deli is an American chain of fast casual restaurants founded in 1989 in Oxford, Mississippi by retired dentist Dr. Don Newcomb.

 There are currently over 400 locations in 26 states, ranging from Virginia in the East to Florida in the South to Arizona in West to Michigan in the North.

 The menu includes deli sandwiches, "Texas-size" spuds (baked potatoes), soups, salads, and desserts, as well as catering items such as sandwich trays and boxed
lunches.

 The chain is also known for its McAlister's Famous Sweet Tea, which is available by the glass or by the gallon.
</item 3>

<item 4>
Panera Bread
Panera Bread Company is an American chain of bakery-café fast casual restaurants in the United States and Canada.

 Its headquarters are in Sunset Hills, Missouri, a suburb of St. Louis, and operates as Saint Louis Bread Company in the St. Louis metropolitan area.

 Offerings include soups, salads, pasta, sandwiches, specialty drinks, and bakery items.
</item 4>

<item 5>
Sanders–Brown Center on Aging
The Sanders–Brown Center on Aging at the University of Kentucky in Lexington, Kentucky, is responsible for research, education and service programs in aging.

 The program was founded in 1963 with the creation of the Council on Aging.

 In 1972, a grant from the Eleanor and John Y. Brown Jr., Foundation led to the construction of the current facility.

 The center was named for the Browns and his then-business partner, (Col.) Harlan Sanders.

 The four-story brick and concrete structure fronts South Limestone at the edge of the medical campus.

 The complex features 32 research laboratories that employs more than 150.
</item 5>

<item 6>
John Y. Brown Jr.
John Young Brown Jr. (born December 28, 1933) is an American politician, entrepreneur, and businessman from the U.S. state of Kentucky.

 He served as the 55th governor of Kentucky from 1979 to 1983, although he may be best known for building Kentucky Fried Chicken (KFC) into a
multimillion-dollar restaurant chain.
</item 6>

<item 7>
VIP's
VIP's, alternatively written Vip's, is a defunct restaurant chain in the Western United States that operated from 1968 until the late 1980s, based in Salem,
Oregon.

 With more than 50 locations, it was once the largest restaurant chain based in Oregon.

 It was a Denny's-style restaurant, a type that was commonly known at that time as a "coffee shop" but is now more commonly known as a casual dining restaurant.

 Most restaurants were located near freeways and were open 24 hours.

 At its peak, the chain had locations in five states: Oregon, Washington, Idaho, Nevada and northern California.
</item 7>

<item 8>
Aladdin's Eatery
Aladdin's Eatery is a chain of franchised restaurants in the American Midwest and South-East, specializing in Lebanese cuisine.

 Adapted to American tastes, the sites are fast casual restaurants that also offer take out.
</item 8>

<item 9>
Rochelle Buffenstein
Rochelle (Shelley) Buffenstein is a staff scientist at Calico, an Alphabet, Inc. funded research outfit investigating aging.

 Previously, she had been a professor of Physiology at theBarshop Institute for Longevity and Aging Studies at the University of Texas Health Science Center at
San Antonio.

 Her research focuses on comparative vertebrate physiology, energetics, cancer biology, and aging.

 She has worked with marsupials, mole-rats, tenrecs, bats, subterranean mammals, and primates.

 Her best known work involves exceptional aging -- specifically, why naked mole-rats live for so much longer than other rodents.

 Her lab has investigated theories of aging including oxidative damage theory, the advanced glycation end product theory, and the telomere theory.
</item 9>

<item 10>
Benjamin A. Rogge
Rogge served for many years as a Liberty Fund trustee.

 Thomas Sowell gives Rogge credit for encouraging him to write a book on economics and race.

 Rogge also was a frequent presenter at the seminars of the Foundation for Economic Education (FEE).

 FEE's founder, Leonard Read, thought of Rogge as Read's eventual successor, an outcome prevented by Read outliving Rogge.

 An extended (but somewhat grainy) video clip of a Rogge FEE lecture on "Competition and Monopoly" on YouTube illustrates the dry wit that made him a popular
speaker.

 Rogge attended 13 meetings of the influential international Mont Pelerin Society.

 Rogge helped produce, and narrated, a documentary on Adam Smith that was funded by Liberty Fund.

 Rogge wrote the introduction to a collection of quotations from Adam Smith.

 A collection of Rogge's speeches, often on topics in economics or education, was published under the title "Can Capitalism Survive?"

 Wabash College, where he taught for many years, established a speaker series in his honor.
</item 10>
=== END ITEM id=d8ce2ae5d7e73271 ===

=== ITEM id=087bf66ae3f5a2b0 ===
Gold sentences and facts:
Sentence 0: The Korean War (in South Korean Hangul: 한국전쟁 ; Hanja: 韓國戰爭 ; RR: "Hanguk Jeonjaeng " , "Korean War"; in North Korean Chosŏn'gŭl: 조국해방전쟁 ; Hancha: 祖國解放戰爭 ; MR: "Choguk haebang chǒnjaeng " , "Fatherland Liberation War"; 25 June 1950 – 27 July 1953) was a war between North Korea (with the support of China and the Soviet Union) and South Korea (with the principal support of the United States).
  Fact 0.0: The Korean War is called 한국전쟁 in South Korean Hangul.
  Fact 0.1: The Korean War is called 韓國戰爭 in South Korean Hanja.
  Fact 0.2: The Korean War is romanized as 'Hanguk Jeonjaeng' (RR), meaning 'Korean War'.
  Fact 0.3: The Korean War is called 조국해방전쟁 in North Korean Chosŏn'gŭl.
  Fact 0.4: The Korean War is called 祖國解放戰爭 in North Korean Hancha.
  Fact 0.5: The Korean War is romanized as 'Choguk haebang chǒnjaeng' (MR), meaning 'Fatherland Liberation War'.
  Fact 0.6: The Korean War lasted from 25 June 1950 to 27 July 1953.
  Fact 0.7: The Korean War was a war between North Korea and South Korea.
  Fact 0.8: North Korea fought with the support of China and the Soviet Union.
  Fact 0.9: South Korea fought with the principal support of the United States.
Sentence 1: The Western Allied invasion of Germany was coordinated by the Western Allies during the final months of hostilities in the European theatre of World War II.
  Fact 1.0: The Western Allied invasion of Germany was coordinated by the Western Allies.
  Fact 1.1: The Western Allied invasion of Germany took place during the final months of hostilities in the European theatre of World War II.
Sentence 2:  The Allied invasion of Germany started with the Western Allies crossing the Rhine River in March 1945 before fanning out and overrunning all of western Germany from the Baltic in the north to Austria in the south before the Germans surrendered on 8 May 1945.
  Fact 2.0: The Allied invasion of Germany started with the Western Allies crossing the Rhine River in March 1945.
  Fact 2.1: After crossing the Rhine, the Western Allies fanned out and overran all of western Germany.
  Fact 2.2: The Western Allies overran western Germany from the Baltic in the north to Austria in the south.
  Fact 2.3: The Germans surrendered on 8 May 1945.

Retrieved items:
<item 1>
Battle of Saipan
The Battle of Saipan was a battle of the Pacific campaign of World War II, fought on the island of Saipan in the Mariana Islands from 15 June to 9 July 1944.

 The Allied invasion fleet embarking the expeditionary forces left Pearl Harbor on 5 June 1944, the day before Operation Overlord in Europe was launched.

 The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division, commanded by Lieutenant General Holland Smith, defeated the 43rd
Infantry Division of the Imperial Japanese Army, commanded by Lieutenant General Yoshitsugu Saito.
</item 1>

<item 2>
Battle of the Ch'ongch'on River
The Battle of the Ch'ongch'on River, also known as the Battle of the Ch'ongch'on () or the Second Phase Campaign Western Sector (), was a decisive battle in the
Korean War, and it took place from November 25 to December 2, 1950, along the Ch'ongch'on River Valley in the northwestern part of North Korea.

 In response to the successful Chinese First Phase Campaign against the United Nations forces, General Douglas MacArthur launched the Home-by-Christmas
Offensive to evict the Chinese forces from Korea and to end the war.

 Anticipating this reaction, the Chinese People's Volunteer Army Commander Peng Dehuai planned a counteroffensive, dubbed the "Second Phase Campaign", against
the advancing UN forces.
</item 2>

<item 3>
Operation Epsom
Operation Epsom, also known as the First Battle of the Odon, was a British Second World War offensive that took place between 26 and 30 June 1944, during the
Battle of Normandy.

 The offensive was intended to outflank and seize the German-occupied city of Caen, an important Allied objective, in the early stages of Operation Overlord,
the Allied invasion of north-west Europe.
</item 3>

<item 4>
Korean War
The Korean War (in South Korean Hangul: 한국전쟁 ; Hanja: 韓國戰爭 ; RR: "Hanguk Jeonjaeng " , "Korean War"; in North Korean Chosŏn'gŭl: 조국해방전쟁 ; Hancha: 祖國解放戰爭 ; MR:
"Choguk haebang chǒnjaeng " , "Fatherland Liberation War"; 25 June 1950 – 27 July 1953) was a war between North Korea (with the support of China and the Soviet
Union) and South Korea (with the principal support of the United States).

 The war began on 25 June 1950 when North Korea invaded South Korea.

 The United Nations, with the United States as the principal force, came to the aid of South Korea.

 China came to the aid of North Korea, and the Soviet Union also gave some assistance to the North.
</item 4>

<item 5>
Schlieffen Plan
The Schlieffen Plan (German: "Schlieffen-Plan" , ] ) was the name given after World War I to the thinking behind the German invasion of France and Belgium on 4
August 1914.

 Field Marshal Alfred von Schlieffen, the Chief of the Imperial Army German General Staff from 1891 to 1906, devised a deployment plan for a war-winning
offensive, in a one-front war against the French Third Republic from 1905–06.

 After the war, the German official historians of the "Reichsarchiv" and other writers, described the plan as a blueprint for victory.

 German historians claimed that the plan had been ruined by "Generaloberst" (Colonel-General) Helmuth von Moltke the Younger, the Commander-in-Chief of the
German army after Schlieffen retired in 1906, who was dismissed after the First Battle of the Marne (5–12 September 1914).
</item 5>

<item 6>
Allied invasion of Italy
The Allied invasion of Italy was the Allied amphibious landing on mainland Italy that took place on 3 September 1943 during the early stages of the Italian
Campaign of World War II.

 The operation was undertaken by General Sir Harold Alexander's 15th Army Group (comprising General Mark W. Clark and General Bernard Montgomery's British
Eighth Army) and followed the successful invasion of Sicily.

 The main invasion force landed around Salerno on 9 September on the western coast in Operation Avalanche, while two supporting operations took place in
Calabria (Operation Baytown) and Taranto (Operation Slapstick).
</item 6>

<item 7>
Battle of Okinawa
The Battle of Okinawa (Japanese: 沖縄戦 , Hepburn: Okinawa-sen ) (Okinawan: 沖縄戦 , "Uchinaa ikusa " ), codenamed Operation Iceberg, was a major battle of the
Pacific War fought on the island of Okinawa by United States Marine and Army forces against the Imperial Japanese Army and included the largest amphibious
assault in the Pacific War during World War II, the April 1, 1945, invasion of Okinawa itself.

 The 82-day battle lasted from April 1 until June 22, 1945.

 After a long campaign of island hopping, the Allies were planning to use Okinawa, a large island only 340 mi away from mainland Japan, as a base for air
operations for the planned invasion of the Japanese home islands.
</item 7>

<item 8>
Battle of Guam (1941)
The First Battle of Guam was an engagement during the Pacific War in World War II, and took place from 8 December to 10 December 1941 on Guam in the Mariana
Islands between Japan and the United States.

 The American garrison was defeated by Japanese forces on 10 December, which resulted in an occupation until the Second Battle of Guam in 1944.
</item 8>

<item 9>
Western Allied invasion of Germany
The Western Allied invasion of Germany was coordinated by the Western Allies during the final months of hostilities in the European theatre of World War II.

 The Allied invasion of Germany started with the Western Allies crossing the Rhine River in March 1945 before fanning out and overrunning all of western Germany
from the Baltic in the north to Austria in the south before the Germans surrendered on 8 May 1945.

 This is known as the "Central Europe Campaign" in United States military histories.
</item 9>

<item 10>
South German Offensive
The South German Offensive is the general name of one of the final offensives of World War II in Europe.

 The offensive was led by the Seventh and Third armies of the United States along with the First Army of France.

 Soviet troops linked up with American forces in Czechoslovakia notably in the Battle of Slivice.

 The offensive was made by the US 6th Army Group to protect the US 12th Army Group's right flank and to prevent a German last stand in the Alps.

 However German resistance was much more fierce than in the north, which slowed the 6th Army Group's progress.

 However, by the end of April, many German divisions surrendered without a fight to the advancing American forces to avoid the inevitable destruction.

 The VI Corps of the Seventh Army linked up with the US Fifth Army, which fought through Italy, in the Alps as the Third Army advanced into Austria and
Czechoslovakia, where it linked up with Soviet forces advancing from the east.

 Fighting continued a few days after the Surrender of Germany on 8 May, due to German forces fighting west to surrender to the Americans instead of the Soviets.
</item 10>
=== END ITEM id=087bf66ae3f5a2b0 ===

=== ITEM id=56bdda227187f542 ===
Gold sentences and facts:
Sentence 0:  Built in 1988 by the TOGO company, it was the fastest roller coaster in the world when it was built (taking the record from American Eagle at Six Flags Great America).
  Fact 0.0: The roller coaster was built in 1988 by the TOGO company.
  Fact 0.1: The roller coaster built by the TOGO company was the fastest roller coaster in the world when it was built.
  Fact 0.2: The roller coaster built by the TOGO company took the record for fastest roller coaster in the world from American Eagle at Six Flags Great America.
Sentence 1: TOGO (株式会社トーゴ , Kabushiki-gaisha Tōgo ) was a Japanese amusement ride company that built roller coasters, giant wheels, carousels, flumes, dark rides, sky cycles and other amusement rides.
  Fact 1.0: TOGO (株式会社トーゴ, Kabushiki-gaisha Tōgo) was a Japanese amusement ride company.
  Fact 1.1: TOGO built roller coasters.
  Fact 1.2: TOGO built giant wheels.
  Fact 1.3: TOGO built carousels.
  Fact 1.4: TOGO built flumes.
  Fact 1.5: TOGO built dark rides.
  Fact 1.6: TOGO built sky cycles.
  Fact 1.7: TOGO built other amusement rides.

Retrieved items:
<item 1>
Castle Park (amusement park)
Castle Park, formerly Castle Amusement Park, is a 25-acre amusement park and family amusement center located in Riverside, California.

 The park utilizes a medieval "castle" theme and includes attractions such as a miniature golf course, arcade, and 27 amusement rides including three roller
coasters such as "Merlin's Revenge", a junior rollercoaster, "Screamin' Demon" a spinning Wild Mouse rollercoaster, and "Little Dipper", a children's
rollercoaster.

 The main "castle" themed building, houses the arcade as well as its only dark ride; "Ghost Blasters", an interactive attraction, designed by Sally Corporation,
which can also be found at other amusement parks throughout North America.

 The park was designed, built and operated by Bud Hurlbut, who designed several rides at Knott's Berry Farm.

 Castle Park is currently owned and operated by Palace Entertainment.
</item 1>

<item 2>
Graph query row:
entity: Bandit
relationship: CREATED_BY
related: TOGO
entity_types: ['__KGBuilder__', '__Entity__', 'Organization']
kind: company
occupation: None
</item 2>

<item 3>
List of roller coaster rankings
Roller coasters are amusement rides developed for amusement parks and modern theme parks.

 During the 16th and 17th centuries, rides consisting of wooden sleds that took riders down large slides made from ice were popular in Russia.

 The first roller coasters, where the train was attached to a wooden track, first appeared in France in the early 1800s.

 Although wooden roller coasters are still being produced, steel roller coasters are more common and can be found on every continent except Antarctica.
</item 3>

<item 4>
Giovanola
Giovanola Freres SA was one of the prominent thrill ride manufacturers in the world.

 It was well known for thrill rides and also built electrical power stations, water storage tanks, pipelines, highway bridges, and many other steel products.

 The company started out as a small metal forging shop, founded by Joseph Giovanola in 1888.

 It served as a subcontractor to Intamin supplying rides and roller coasters.

 In 1998 Giovanola started marketing directly under the name Giovanola Amusement Rides Worldwide.

 The company was based in Monthey, Switzerland.

 During its last years in business, Giovanola fashioned steel behind the scenes for companies such as Intamin and Bolliger & Mabillard.

 The company also built its own thrill rides and roller coasters from 1998 to 2001.

 Examples of Giovanola roller coasters are Goliath at Six Flags Magic Mountain, Titan at Six Flags Over Texas and Anaconda at South Africa's Gold Reef City.
</item 4>

<item 5>
Pretzel Amusement Ride Company
The Pretzel Amusement Ride Company was a famous manufacturer of pretzel dark rides. Pretzel built over 1400 pretzel rides and sold them to carnivals and parks.

 The pretzel ride was invented by Marvin Rempfer.

 Leon Cassidy was Marvin's partner in the Pretzel Amusement Ride Company.

 Both names are on the patent.

 Leon Cassidy patented the single-rail dark ride in 1928 along with Marvin Rempfer.

 The company originated in Tumbling Dam Park on the banks of Sunset Lake in Bridgeton, New Jersey.

 A rider said that ""It felt like I was turned and twisted like a Pretzel"", so the name Pretzel was chosen.

 A large heavy pretzel design was originally affixed to the front of each car to prevent the car from flipping backwards.

 In 1929, a standard Pretzel ride had five cars, 350 feet of track, and was one and a half minutes per ride.

 A pretzel ride sold for $1,200.
</item 5>

<item 6>
Anton Schwarzkopf
Anton Schwarzkopf (8 July 1924 – 30 July 2001) was a German engineer of amusement rides, and founder of the Schwarzkopf Industries Company, which built numerous
amusement rides and large roller coasters for both amusement parks and traveling funfairs.
</item 6>

<item 7>
Millennium Force
Millennium Force is a steel roller coaster built by Intamin at Cedar Point amusement park in Sandusky, Ohio, United States.

 It was the fourteenth roller coaster to be built at the park since Blue Streak opened in 1964.

 Upon completion in 2000, Millennium Force broke six world records and was the world's first Giga Coaster, a roller coaster that exceeds 300 ft in height and
completes a full circuit.

 It was briefly the tallest and fastest in the world until Steel Dragon 2000 opened later the same year.

 The ride is also the third-longest roller coaster in North America after The Beast at Kings Island and Fury 325 at Carowinds.

 It was the first roller coaster to use a cable lift system rather than a traditional chain lift.

 The coaster has a 310 ft , 45-degree lift hill with a 300 ft drop and features two tunnels, three overbanked turns, and four hills.

 It has a top speed of 93 mph .
</item 7>

<item 8>
Bandit (Yomiuriland)
Bandit (バンデット , Bandetto ) is a steel roller coaster located at Yomiuriland in the city of Inagi, near Tokyo, Japan.

 Built in 1988 by the TOGO company, it was the fastest roller coaster in the world when it was built (taking the record from American Eagle at Six Flags Great
America).

 It lost the record to Magnum XL-200 at Cedar Point one year later.
</item 8>

<item 9>
Mack Rides
Mack Rides GmbH & Co KG, also known simply as Mack Rides, is a German company that designs and constructs amusement rides.

 Mack Rides in headquartered in Waldkirch, Baden-Württemberg, Germany.

 Mack Rides is one of the world's oldest amusement industry suppliers and builds all kinds of amusement devices including several types of flat rides, dark
rides, log flumes, tow boat rides and roller coasters.

 The family that owns Mack Rides also owns Europa-Park.
</item 9>

<item 10>
TOGO
TOGO (株式会社トーゴ , Kabushiki-gaisha Tōgo ) was a Japanese amusement ride company that built roller coasters, giant wheels, carousels, flumes, dark rides, sky
cycles and other amusement rides.
</item 10>
=== END ITEM id=56bdda227187f542 ===
