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

This file has 5 items.

=== ITEM id=d06fa24a84e001fa ===
Gold sentences and facts:
Sentence 0: When following the western side of the African coastline from the equator , however , the Cape of Good Hope marks the point where a ship begins to travel more eastward than southward . Thus , the first modern rounding of the cape in 1488 by Portuguese explorer Bartolomeu Dias was a milestone in the attempts by the Portuguese to establish direct trade relations with the Far East ( although Herodotus mentioned a claim that the Phoenicians had done so far earlier ) . Dias called the cape Cabo das Tormentas ( `` Cape of Storms '' ; Dutch : Stormkaap ) , which was the original name of the `` Cape of Good Hope '' .
  Fact 0.0: When following the western side of the African coastline from the equator, the Cape of Good Hope marks the point where a ship begins to travel more eastward than southward.
  Fact 0.1: Portuguese explorer Bartolomeu Dias made the first modern rounding of the Cape of Good Hope in 1488.
  Fact 0.2: The 1488 rounding of the cape by Bartolomeu Dias was a milestone in the attempts by the Portuguese to establish direct trade relations with the Far East.
  Fact 0.3: Herodotus mentioned a claim that the Phoenicians had rounded the cape far earlier.
  Fact 0.4: Bartolomeu Dias called the cape Cabo das Tormentas.
  Fact 0.5: Cabo das Tormentas means 'Cape of Storms' in English.
  Fact 0.6: The Dutch name for Cabo das Tormentas is Stormkaap.
  Fact 0.7: Cabo das Tormentas was the original name of the Cape of Good Hope.

Retrieved items:
<item 1>
Cape of Good Hope
Contents ( hide ) 1 History 2 Geography 3 Fauna 3.1 Chacma baboons 4 Flora 5 Legends 6 See also 7 References 8 External links History ( edit ) Eudoxus of
Cyzicus ( / ˈjuːdəksəs / ; Greek : Εὔδοξος , Eúdoxos ; fl . c. 130 BC ) was a Greek navigator for Ptolemy VIII , king of the Hellenistic Ptolemaic dynasty in
Egypt , who found the wreck of a ship in the Indian Ocean that appeared to have come from Gades ( today 's Cádiz in Spain ) , rounding the Cape . When Eudoxus
was returning from his second voyage to India the wind forced him south of the Gulf of Aden and down the coast of Africa for some distance . Somewhere along the
coast of East Africa , he found the remains of the ship .
</item 1>

<item 2>
Cape of Good Hope
Fra Mauro puts the following inscription by the southern tip of Africa , which he names the `` Cape of Diab '' , describing the exploration by a ship from the
East around 1420 : Detail of the Fra Mauro Map describing the construction of the junks that navigate in the Indian Ocean . `` Around 1420 a ship , or junk ,
from India crossed the Sea of India towards the Island of Men and the Island of Women , off Cape Diab , between the Green Islands and the shadows . It sailed
for 40 days in a south - westerly direction without ever finding anything other than wind and water . According to these people themselves , the ship went some
2,000 miles ahead until - once favourable conditions came to an end - it turned round and sailed back to Cape Diab in 70 days '' . `` The ships called junks (
lit . `` Zonchi '' ) that navigate these seas carry four masts or more , some of which can be raised or lowered , and have 40 to 60 cabins for the merchants and
only one tiller .
</item 2>

<item 3>
Cape of Good Hope
In the Early Modern Era , the first European to reach the cape was the Portuguese explorer Bartolomeu Dias on 12 March 1488 , who named it the `` Cape of Storms
'' ( Cabo das Tormentas ) . It was later renamed by John II of Portugal as `` Cape of Good Hope '' ( Cabo da Boa Esperança ) because of the great optimism
engendered by the opening of a sea route to India and the East . The land around the cape was home to the Khoikhoi people when the Dutch first settled there in
1652 . The Khoikhoi had arrived in these parts about fifteen hundred years before . They were called Hottentots by the Dutch , a term that has now come to be
regarded as pejorative . Dutch colonial administrator Jan van Riebeeck established a resupply camp for the Dutch East India Company some 50 km north of the cape
in Table Bay on 6 April 1652 and this eventually developed into Cape Town . Supplies of fresh food were vital on the long journey around Africa and Cape Town
became known as `` The Tavern of the Seas '' .
</item 3>

<item 4>
Cape of Good Hope
Somewhere along the coast of East Africa , he found the remains of the ship . Due to its appearance and the story told by the natives , Eudoxus concluded that
the ship was from Gades and had sailed anti-clockwise around Africa , passing the Cape and entering the Indian Ocean . This inspired him to repeat the voyage
and attempt a circumnavigation of the continent . Organising the expedition on his own account he set sail from Gades and began to work down the African coast .
The difficulties were too great , however , and he was obliged to return to Europe . After this failure he again set out to circumnavigate Africa . His eventual
fate is unknown . Although some , such as Pliny , claimed that Eudoxus did achieve his goal , the most probable conclusion is that he perished on the journey .
In the 1450 Fra Mauro map , the Indian Ocean is depicted as connected to the Atlantic .
</item 4>

<item 5>
Trade winds
They also learned that to reach South Africa , they needed to go far out in the ocean , head for Brazil , and around 30 ° S go east again . Following the
African coast southbound means upwind in the Southern hemisphere . In the Pacific ocean , the full wind circulation , which included both the trade wind
easterlies and higher - latitude Westerlies , was unknown to Europeans until Andres de Urdaneta 's voyage in 1565 . The captain of a sailing ship seeks a course
along which the winds can be expected to blow in the direction of travel . During the Age of Sail , the pattern of prevailing winds made various points of the
globe easy or difficult to access , and therefore had a direct effect on European empire - building and thus on modern political geography . For example ,
Manila galleons could not sail into the wind at all . By the 18th century the importance of the trade winds to England 's merchant fleet for crossing the
Atlantic Ocean had led both the general public and etymologists to identify the name with a later meaning of ' trade ' , `` ( foreign ) commerce '' .
</item 5>

<item 6>
Cape of Good Hope
Fra Mauro also comments that the account of the expedition , together with the relation by Strabo of the travels of Eudoxus of Cyzicus from Arabia to Gibraltar
through the southern Ocean in Antiquity , led him to believe that the Indian Ocean was not a closed sea and that Africa could be circumnavigated by her southern
end ( Text from Fra Mauro map , 11 , G2 ) . This knowledge , together with the map depiction of the African continent , probably encouraged the Portuguese to
intensify their effort to round the tip of Africa . Map showing the Cape Peninsula , illustrating the position of the Cape of Good Hope . The main mountains and
their peaks , including Table Mountain , and its relation to the City of Cape Town are shown . Map of the Cape of Good Hope and Cape Agulhas the southernmost
point of Africa . Main article : Cape Colony Reproduction of the Cross of Vasco da Gama at the Cape of Good Hope .
</item 6>

<item 7>
Cape of Good Hope
Legends ( edit ) The Cape of Good Hope is the legendary home of The Flying Dutchman . Crewed by tormented and damned ghostly sailors , it is doomed forever to
beat its way through the adjacent waters without ever succeeding in rounding the headland . Adamastor is a Greek - type mythological character invented by the
Portuguese poet Luís de Camões in his epic poem Os Lusíadas ( first printed in 1572 ) , as a symbol of the forces of nature Portuguese navigators had to
overcome during their discoveries and more specifically of the dangers Portuguese sailors faced when trying to round the Cape of Storms . See also ( edit ) Cape
Peninsula Cape Horn , its South American counterpart Cape Point Western Cape History of Cape Colony Simon 's Town Postage stamps and postal history of the Cape
of Good Hope References ( edit ) Jump up ^ Every word in isolation ( ˌkaːp də ˌɣudə ˈɦoːp ) . Jump up ^ `` Cape of Good Hope , South Africa - 360 ° Aerial
Panoramas '' . Jump up ^ The first circumnavigation of Africa .
</item 7>

<item 8>
Cape of Good Hope
Contemporary scientific knowledge instead states the southernmost point of Africa is Cape Agulhas about 150 kilometres ( 90 mi ) to the east - southeast . The
currents of the two oceans meet at the point where the warm - water Agulhas current meets the cold water Benguela current and turns back on itself . That
oceanic meeting point fluctuates between Cape Agulhas and Cape Point ( about 1.2 kilometres ( 0.75 mi ) east of the Cape of Good Hope ) . When following the
western side of the African coastline from the equator , however , the Cape of Good Hope marks the point where a ship begins to travel more eastward than
southward . Thus , the first modern rounding of the cape in 1488 by Portuguese explorer Bartolomeu Dias was a milestone in the attempts by the Portuguese to
establish direct trade relations with the Far East ( although Herodotus mentioned a claim that the Phoenicians had done so far earlier ) . Dias called the cape
Cabo das Tormentas ( `` Cape of Storms '' ; Dutch : Stormkaap ) , which was the original name of the `` Cape of Good Hope '' .
</item 8>

<item 9>
Cape of Good Hope
They can navigate without a compass , because they have an astrologer , who stands on the side and , with an astrolabe in hand , gives orders to the navigator
`` . ( Text from the Fra Mauro map , 09 - P25 . ) Fra Mauro explained that he obtained the information from `` a trustworthy source '' , who traveled with the
expedition , possibly the Venetian explorer Niccolò da Conti who happened to be in Calicut , India at the time the expedition left : `` What is more , I have
spoken with a person worthy of trust , who says that he sailed in an Indian ship caught in the fury of a tempest for 40 days out in the Sea of India , beyond
the Cape of Soffala and the Green Islands towards west - southwest ; and according to the astrologers who act as their guides , they had advanced almost 2,000
miles . Thus one can believe and confirm what is said by both these and those , and that they had therefore sailed 4,000 miles '' .
</item 9>

<item 10>
Cape of Good Hope
Jump up ^ The first circumnavigation of Africa . livius.org Jump up ^ Sarah Mytton Maury ( 1848 ) . Englishwoman In America . p. 33 . Jump up ^ Along the
Clipper Way , Francis Chichester ; page 78 . Hodder & Stoughton , 1966 . ISBN 0 - 340 - 00191 - 7 Jump up ^ Map of the Park , showing the Cape of Good Hope
section ( retrieved 27 March 2010 ) . Jump up ^ Tozer , Henry F. ( 1997 ) . History of Ancient Geography . Biblo & Tannen . pp. 189 -- 190 . ISBN 0 - 8196 -
0138 - 1 . ; online at Google Books Jump up ^ Tozer , Henry F. ( 1997 ) . History of Ancient Geography . Biblo & Tannen . pp. xxiii . ISBN 0 - 8196 - 0138 - 1 .
; online at Google Books Jump up ^ Marco Polo , p.
</item 10>
=== END ITEM id=d06fa24a84e001fa ===

=== ITEM id=c9a095ef47ab65f6 ===
Gold sentences and facts:
Sentence 0: When following the western side of the African coastline from the equator , however , the Cape of Good Hope marks the point where a ship begins to travel more eastward than southward . Thus , the first modern rounding of the cape in 1488 by Portuguese explorer Bartolomeu Dias was a milestone in the attempts by the Portuguese to establish direct trade relations with the Far East ( although Herodotus mentioned a claim that the Phoenicians had done so far earlier ) . Dias called the cape Cabo das Tormentas ( `` Cape of Storms '' ; Dutch : Stormkaap ) , which was the original name of the `` Cape of Good Hope '' .
  Fact 0.0: When following the western side of the African coastline from the equator, the Cape of Good Hope marks the point where a ship begins to travel more eastward than southward.
  Fact 0.1: Portuguese explorer Bartolomeu Dias made the first modern rounding of the Cape of Good Hope in 1488.
  Fact 0.2: The 1488 rounding of the cape by Bartolomeu Dias was a milestone in the attempts by the Portuguese to establish direct trade relations with the Far East.
  Fact 0.3: Herodotus mentioned a claim that the Phoenicians had rounded the cape far earlier.
  Fact 0.4: Bartolomeu Dias called the cape Cabo das Tormentas.
  Fact 0.5: Cabo das Tormentas means 'Cape of Storms' in English.
  Fact 0.6: The Dutch name for Cabo das Tormentas is Stormkaap.
  Fact 0.7: Cabo das Tormentas was the original name of the Cape of Good Hope.

Retrieved items:
<item 1>
Cape of Good Hope
Fra Mauro also comments that the account of the expedition , together with the relation by Strabo of the travels of Eudoxus of Cyzicus from Arabia to Gibraltar
through the southern Ocean in Antiquity , led him to believe that the Indian Ocean was not a closed sea and that Africa could be circumnavigated by her southern
end ( Text from Fra Mauro map , 11 , G2 ) . This knowledge , together with the map depiction of the African continent , probably encouraged the Portuguese to
intensify their effort to round the tip of Africa . Map showing the Cape Peninsula , illustrating the position of the Cape of Good Hope . The main mountains and
their peaks , including Table Mountain , and its relation to the City of Cape Town are shown . Map of the Cape of Good Hope and Cape Agulhas the southernmost
point of Africa . Main article : Cape Colony Reproduction of the Cross of Vasco da Gama at the Cape of Good Hope .
</item 1>

<item 2>
Cape of Good Hope
Fra Mauro puts the following inscription by the southern tip of Africa , which he names the `` Cape of Diab '' , describing the exploration by a ship from the
East around 1420 : Detail of the Fra Mauro Map describing the construction of the junks that navigate in the Indian Ocean . `` Around 1420 a ship , or junk ,
from India crossed the Sea of India towards the Island of Men and the Island of Women , off Cape Diab , between the Green Islands and the shadows . It sailed
for 40 days in a south - westerly direction without ever finding anything other than wind and water . According to these people themselves , the ship went some
2,000 miles ahead until - once favourable conditions came to an end - it turned round and sailed back to Cape Diab in 70 days '' . `` The ships called junks (
lit . `` Zonchi '' ) that navigate these seas carry four masts or more , some of which can be raised or lowered , and have 40 to 60 cabins for the merchants and
only one tiller .
</item 2>

<item 3>
Cape of Good Hope
They can navigate without a compass , because they have an astrologer , who stands on the side and , with an astrolabe in hand , gives orders to the navigator
`` . ( Text from the Fra Mauro map , 09 - P25 . ) Fra Mauro explained that he obtained the information from `` a trustworthy source '' , who traveled with the
expedition , possibly the Venetian explorer Niccolò da Conti who happened to be in Calicut , India at the time the expedition left : `` What is more , I have
spoken with a person worthy of trust , who says that he sailed in an Indian ship caught in the fury of a tempest for 40 days out in the Sea of India , beyond
the Cape of Soffala and the Green Islands towards west - southwest ; and according to the astrologers who act as their guides , they had advanced almost 2,000
miles . Thus one can believe and confirm what is said by both these and those , and that they had therefore sailed 4,000 miles '' .
</item 3>

<item 4>
Cape of Good Hope
In the Early Modern Era , the first European to reach the cape was the Portuguese explorer Bartolomeu Dias on 12 March 1488 , who named it the `` Cape of Storms
'' ( Cabo das Tormentas ) . It was later renamed by John II of Portugal as `` Cape of Good Hope '' ( Cabo da Boa Esperança ) because of the great optimism
engendered by the opening of a sea route to India and the East . The land around the cape was home to the Khoikhoi people when the Dutch first settled there in
1652 . The Khoikhoi had arrived in these parts about fifteen hundred years before . They were called Hottentots by the Dutch , a term that has now come to be
regarded as pejorative . Dutch colonial administrator Jan van Riebeeck established a resupply camp for the Dutch East India Company some 50 km north of the cape
in Table Bay on 6 April 1652 and this eventually developed into Cape Town . Supplies of fresh food were vital on the long journey around Africa and Cape Town
became known as `` The Tavern of the Seas '' .
</item 4>

<item 5>
Cape of Good Hope
Contemporary scientific knowledge instead states the southernmost point of Africa is Cape Agulhas about 150 kilometres ( 90 mi ) to the east - southeast . The
currents of the two oceans meet at the point where the warm - water Agulhas current meets the cold water Benguela current and turns back on itself . That
oceanic meeting point fluctuates between Cape Agulhas and Cape Point ( about 1.2 kilometres ( 0.75 mi ) east of the Cape of Good Hope ) . When following the
western side of the African coastline from the equator , however , the Cape of Good Hope marks the point where a ship begins to travel more eastward than
southward . Thus , the first modern rounding of the cape in 1488 by Portuguese explorer Bartolomeu Dias was a milestone in the attempts by the Portuguese to
establish direct trade relations with the Far East ( although Herodotus mentioned a claim that the Phoenicians had done so far earlier ) . Dias called the cape
Cabo das Tormentas ( `` Cape of Storms '' ; Dutch : Stormkaap ) , which was the original name of the `` Cape of Good Hope '' .
</item 5>

<item 6>
Cape of Good Hope
Jump up ^ The first circumnavigation of Africa . livius.org Jump up ^ Sarah Mytton Maury ( 1848 ) . Englishwoman In America . p. 33 . Jump up ^ Along the
Clipper Way , Francis Chichester ; page 78 . Hodder & Stoughton , 1966 . ISBN 0 - 340 - 00191 - 7 Jump up ^ Map of the Park , showing the Cape of Good Hope
section ( retrieved 27 March 2010 ) . Jump up ^ Tozer , Henry F. ( 1997 ) . History of Ancient Geography . Biblo & Tannen . pp. 189 -- 190 . ISBN 0 - 8196 -
0138 - 1 . ; online at Google Books Jump up ^ Tozer , Henry F. ( 1997 ) . History of Ancient Geography . Biblo & Tannen . pp. xxiii . ISBN 0 - 8196 - 0138 - 1 .
; online at Google Books Jump up ^ Marco Polo , p.
</item 6>

<item 7>
Cape of Good Hope
It remained a British colony until being incorporated into the independent Union of South Africa in 1910 ( now known as the Republic of South Africa ) . The
Portuguese government erected two navigational beacons , Dias Cross and Gama Cross , to commemorate Vasco da Gama and Bartolomeu Dias as explorers who as
mentioned were the first explorers to reach the cape . When lined up , the crosses point to Whittle Rock ( 34 ° 21 ′ 24.63 '' S 18 ° 28 ′ 26.36 '' E ﻿ / ﻿
34.3568417 ° S 18.4739889 ° E ﻿ / - 34.3568417 ; 18.4739889 ) , a large , permanently submerged shipping hazard in False Bay . Two other beacons in Simon 's
Town provide the intersection . Geography ( edit ) Sign at the Cape of Good Hope The Cape of Good Hope is at the southern tip of the Cape Peninsula , about 2.3
kilometres ( 1.4 mi ) west and a little south of Cape Point on the south - east corner .
</item 7>

<item 8>
Cape of Good Hope
Legends ( edit ) The Cape of Good Hope is the legendary home of The Flying Dutchman . Crewed by tormented and damned ghostly sailors , it is doomed forever to
beat its way through the adjacent waters without ever succeeding in rounding the headland . Adamastor is a Greek - type mythological character invented by the
Portuguese poet Luís de Camões in his epic poem Os Lusíadas ( first printed in 1572 ) , as a symbol of the forces of nature Portuguese navigators had to
overcome during their discoveries and more specifically of the dangers Portuguese sailors faced when trying to round the Cape of Storms . See also ( edit ) Cape
Peninsula Cape Horn , its South American counterpart Cape Point Western Cape History of Cape Colony Simon 's Town Postage stamps and postal history of the Cape
of Good Hope References ( edit ) Jump up ^ Every word in isolation ( ˌkaːp də ˌɣudə ˈɦoːp ) . Jump up ^ `` Cape of Good Hope , South Africa - 360 ° Aerial
Panoramas '' . Jump up ^ The first circumnavigation of Africa .
</item 8>

<item 9>
Graph query row:
person: Bartolomeu Dias
relationship: PARTICIPATED_IN
connected_entity: first modern rounding of the cape
date: 1488
role: None
fact: None
context: None
</item 9>

<item 10>
Cape of Good Hope
Contents ( hide ) 1 History 2 Geography 3 Fauna 3.1 Chacma baboons 4 Flora 5 Legends 6 See also 7 References 8 External links History ( edit ) Eudoxus of
Cyzicus ( / ˈjuːdəksəs / ; Greek : Εὔδοξος , Eúdoxos ; fl . c. 130 BC ) was a Greek navigator for Ptolemy VIII , king of the Hellenistic Ptolemaic dynasty in
Egypt , who found the wreck of a ship in the Indian Ocean that appeared to have come from Gades ( today 's Cádiz in Spain ) , rounding the Cape . When Eudoxus
was returning from his second voyage to India the wind forced him south of the Gulf of Aden and down the coast of Africa for some distance . Somewhere along the
coast of East Africa , he found the remains of the ship .
</item 10>
=== END ITEM id=c9a095ef47ab65f6 ===

=== ITEM id=650566f46dc0626c ===
Gold sentences and facts:
Sentence 0: `` Ca n't Get You Out of My Head '' is a song recorded by Australian singer Kylie Minogue for her eighth studio album , titled Fever , which she released in 2001 . The song was released in Australia by Parlophone as the lead single from the album on 8 September 2001 . It was released on 17 September 2001 in the United Kingdom . In the United States , the single was released on 18 February 2002 . Jointly written , composed , and produced by Cathy Dennis and Rob Davis , `` Ca n't Get You Out of My Head '' is a midtempo dance - pop song which lyrically details its narrator 's obsession towards her lover . The song is famous for its `` la la la '' hook .
  Fact 0.0: 'Can't Get You Out of My Head' is a song.
  Fact 0.1: 'Can't Get You Out of My Head' is recorded by Australian singer Kylie Minogue.
  Fact 0.2: 'Can't Get You Out of My Head' was recorded for Kylie Minogue's eighth studio album.
  Fact 0.3: The eighth studio album is titled Fever.
  Fact 0.4: Kylie Minogue released Fever in 2001.
  Fact 0.5: 'Can't Get You Out of My Head' was released in Australia by Parlophone.
  Fact 0.6: 'Can't Get You Out of My Head' was released in Australia as the lead single from Fever.
  Fact 0.7: The Australian release of 'Can't Get You Out of My Head' was on 8 September 2001.
  Fact 0.8: 'Can't Get You Out of My Head' was released in the United Kingdom on 17 September 2001.
  Fact 0.9: 'Can't Get You Out of My Head' was released in the United States as a single on 18 February 2002.
  Fact 0.10: 'Can't Get You Out of My Head' was jointly written by Cathy Dennis and Rob Davis.
  Fact 0.11: 'Can't Get You Out of My Head' was jointly composed by Cathy Dennis and Rob Davis.
  Fact 0.12: 'Can't Get You Out of My Head' was jointly produced by Cathy Dennis and Rob Davis.
  Fact 0.13: 'Can't Get You Out of My Head' is a midtempo dance-pop song.
  Fact 0.14: 'Can't Get You Out of My Head' lyrically details its narrator's obsession towards her lover.
  Fact 0.15: The song is famous for its 'la la la' hook.

Retrieved items:
<item 1>
Can't Get You Out of My Head
By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit
organization . About Wikipedia
</item 1>

<item 2>
Can't Get You Out of My Head
The song was recorded using Cubase music software , which Davis ran on his Mac computer . Davis began playing an acoustic guitar and ran a 125 beats per minute
drum loop , on which Dennis , herself a singer who had enjoyed chart success in the United States , began singing the line `` I just ca n't get you out of my
head '' in the key of D minor . After three and a half hours , the demo was recorded and the vocals were laid afterwards . Dennis called their recording setup
for the song `` the most primitive set - up you could imagine ! Different producers work in different ways . But it 's good to be reminded you do n't have to be
reliant on equipment . A song is about melody and lyrics and being able to take something away in your memory that is going to haunt you . '' She also regarded
their production as a `` very natural and fluid process , '' saying : `` We know how hard we work sometimes to write songs and then spend months picking them to
pieces , but this was the easiest process , the chemicals were all happy and working together . ''
</item 2>

<item 3>
Can't Get You Out of My Head
Through the lyrics of the song , its singer expresses an obsession with an anonymous figure . Dorian Lynskey from The Guardian termed the song a `` mystery ''
as Minogue never reveals the identity of her object of infatuation . The critic suggested that the person Minogue is referring to is either `` a partner , an
evasive one - night stand or someone who does n't know she exists . '' Writing for the same newspaper , Everett True identified a `` darker element '' in the
simple lyrics and felt this sentiment was echoed in Minogue 's restrained vocals . Further , True emphasised that while Minogue 's 1987 single `` I Should Be So
Lucky '' had presented an optimistic romantic future , `` Ca n't Get You Out of My Head '' focuses on an `` unhealthy '' and potentially destructive obsession .
He also noted that in the former song , Minogue played `` the wide - eyed ingénue with alacrity , '' but in the latter song she is aware of the harmful nature
of her infatuation , calling it a `` desire that is wholly dependent on her own self - control . ''
</item 3>

<item 4>
Can't Get You Out of My Head
The music , excluding the guitar part , was programmed using Korg Triton workstation via MIDI . Dennis later remarked : `` Even though Kylie was n't the first
artist to be offered the song , I do n't believe it was meant to go to anyone other than Kylie , and I do n't believe anyone else would have done the incredible
job she did with it , with the video , looking super-hot ! '' Composition ( edit ) `` Ca n't Get You Out of My Head '' A 21 - second sample of `` Ca n't Get You
Out of My Head '' containing the hook and part of the chorus of the song . The song is known for its `` la la la '' hook . Problems playing this file ? See
media help . `` Ca n't Get You Out of My Head '' is a `` robotic '' midtempo dance - pop song , with a tempo of 126 beats per minute .
</item 4>

<item 5>
Can't Get You Out of My Head
My A&R at the time , Miles Leonard and Jamie Nelson , said , `` We 've got something . Come into the office . We 've got to play you something . '' So I went to
the EMI office , and I had the same reaction you did , about 20 seconds in . I could n't even fathom what I was hearing . It just ... did something . I was
beside myself . Then at the end of the song , panic set in . I was saying `` Are you sure we 've got this song ? Do n't tell me that we do n't ! Is it secured ?
Can we have it ? '' And we did ! And that kick - started a whole different phase in my career . -- Minogue , about her reaction to hearing the demo of the song
. `` Ca n't Get You Out of My Head '' was jointly written , composed , and produced by Cathy Dennis and Rob Davis . Dennis and Davis had been brought together
by British artist manager Simon Fuller , who wanted the duo to come up with a song for British pop group S Club 7 .
</item 5>

<item 6>
Can't Get You Out of My Head
It was released on 17 September 2001 in the United Kingdom . In the United States , the single was released on 18 February 2002 . Jointly written , composed ,
and produced by Cathy Dennis and Rob Davis , `` Ca n't Get You Out of My Head '' is a midtempo dance - pop song which lyrically details its narrator 's
obsession towards her lover . The song is famous for its `` la la la '' hook . In addition to positive reviews from music critics , `` Ca n't Get You Out of My
Head '' found commercial success on a large scale . It peaked at number one on the charts of Austria , Belgium , France , Germany , Italy , Poland , Switzerland
, the United Kingdom , and every other European country excluding Finland . It also topped the charts of Minogue 's native country Australia , Canada and New
Zealand . In the United States , the song peaked at number seven on the US Billboard Hot 100 chart , becoming Minogue 's biggest hit in the region since `` The
Loco - Motion '' . `` Ca n't Get You Out of My Head '' reportedly reached number one in 40 countries across the globe .
</item 6>

<item 7>
Can't Get You Out of My Head
Ca n't Get You Out of My Head - Wikipedia Ca n't Get You Out of My Head Jump to : navigation , search `` Ca n't Get You Out of My Head '' Single cover used for
CD # 02 and digital releases of the song .
</item 7>

<item 8>
Can't Get You Out of My Head
: 26 Digital download `` Ca n't Get You Out of My Head '' -- 3 : 52 `` Boy '' -- 3 : 48 Charts ( edit ) Weekly charts ( edit ) Chart ( 2001 -- 02 ) Peak
position Australia ( ARIA ) Austria ( Ö3 Austria Top 40 ) Belgium ( Ultratop 50 Flanders ) Belgium ( Ultratop 50 Wallonia ) Brazil ( ABPD ) Canada ( Canadian
Hot 100 ) Denmark ( Tracklisten ) Finland ( Suomen virallinen lista ) 5 France ( SNEP ) Germany ( Official German Charts ) Greece ( IFPI Greece ) Ireland ( IRMA
) Italy ( FIMI ) Netherlands ( Single Top 100 ) New Zealand ( Recorded Music NZ ) Norway ( VG - lista ) Poland ( Polish Singles Chart ) Portugal ( Billboard )
Romania ( Romanian Top 100 ) Scotland ( Official Charts Company ) South Africa ( RiSA ) Spain ( PROMUSICAE ) Sweden ( Sverigetopplistan ) Switzerland (
Schweizer Hitparade ) UK Singles ( Official Charts Company ) US Billboard Hot 100 7 US Adult Top 40 ( Billboard ) 23 US Dance Club Songs ( Billboard )
</item 8>

<item 9>
Can't Get You Out of My Head
In 2012 , Priya Elan from NME ranked the song at number four on his `` The Greatest Pop Songs in History '' list , saying : `` It was unlike any song I remember
hearing before . '' In 2012 , The Guardian included the song in their list of `` The Best Number One Records , '' labelling it to be `` sleek , Arctic - blue
minimalism , like an emotionally thwarted retelling of Donna Summer 's `` I Feel Love . '' In the same year , PRS for Music , a UK copyright collection society
and performance rights organisation which collects royalties on behalf of songwriters and composers , named `` Ca n't Get You Out of My Head '' the `` Most
Popular Song of the Decade '' as it received the highest airplay and live covers in the 2000s ( decade ) . Minogue performing `` Ca n't Get You Out of My Head
'' during her KylieX Tour ( 2008 ) .
</item 9>

<item 10>
Can't Get You Out of My Head
In 2013 , a survey of 700 people was conducted as part of the Manchester Science Festival to find out which song they considered the `` catchiest , '' and `` Ca
n't Get You Out of My Head '' topped the poll . Lee Barron , in his essay The Seven Ages of Kylie Minogue : Postmodernism , Identity , and Performative Mimicry
, noted that the song `` further established Minogue 's cultural and commercial relevance in the new millennium . '' He remarked that the song `` with its
hypnotic ' la la la ' refrain and the deceptively uncomplicated , catchily repetitive beats and synth - sound , marked yet another clearly defined image
transformation from the camp - infused Light Years to an emphasis upon a cool , machine - like sexuality , a trait clearly identifiable within the promotional
video for ' Ca n't Get You Out of My Head . '
</item 10>
=== END ITEM id=650566f46dc0626c ===

=== ITEM id=37ab120d41e2c9ad ===
Gold sentences and facts:
Sentence 0: Costs are associated with particular goods using one of the several formulas , including specific identification , first - in first - out ( FIFO ) , or average cost . Costs include all costs of purchase , costs of conversion and other costs that are incurred in bringing the inventories to their present location and condition . Costs of goods made by the businesses include material , labor , and allocated overhead . The costs of those goods which are not yet sold are deferred as costs of inventory until the inventory is sold or written down in value .
  Fact 0.0: Costs are associated with particular goods using one of several formulas.
  Fact 0.1: The formulas include specific identification.
  Fact 0.2: The formulas include first-in first-out (FIFO).
  Fact 0.3: The formulas include average cost.
  Fact 0.4: Costs include all costs of purchase.
  Fact 0.5: Costs include all costs of conversion.
  Fact 0.6: Costs include other costs that are incurred in bringing the inventories to their present location and condition.
  Fact 0.7: Costs of goods made by businesses include material.
  Fact 0.8: Costs of goods made by businesses include labor.
  Fact 0.9: Costs of goods made by businesses include allocated overhead.
  Fact 0.10: Costs of goods which are not yet sold are deferred as costs of inventory.
  Fact 0.11: Costs of goods which are not yet sold are deferred until the inventory is sold.
  Fact 0.12: Costs of goods which are not yet sold are deferred until the inventory is written down in value.

Retrieved items:
<item 1>
Cost of goods sold
Most businesses make more than one of a particular item . Thus , costs are incurred for multiple items rather than a particular item sold . Determining how much
of each of these components to allocate to particular goods requires either tracking the particular costs or making some allocations of costs . Parts and raw
materials are often tracked to particular sets ( e.g. , batches or production runs ) of goods , then allocated to each item . Labor costs include direct labor
and indirect labor . Direct labor costs are the wages paid to those employees who spend all their time working directly on the product being manufactured .
Indirect labor costs are the wages paid to other factory employees involved in production . Costs of payroll taxes and fringe benefits are generally included in
labor costs , but may be treated as overhead costs . Labor costs may be allocated to an item or set of items based on timekeeping records . Costs of materials
include direct raw materials , as well as supplies and indirect materials . Where non-incidental amounts of supplies are maintained , the taxpayer must keep
inventories of the supplies for income tax purposes , charging them to expense or cost of goods sold as used rather than as purchased .
</item 1>

<item 2>
Cost of goods sold
Other methods may be used to associate overhead costs with particular goods produced . Overhead rates may be standard rates , in which case there may be
variances , or may be adjusted for each set of goods produced . Identification conventions ( edit ) In some cases , the cost of goods sold may be identified
with the item sold . Ordinarily , however , the identity of goods is lost between the time of purchase or manufacture and the time of sale . Determining which
goods have been sold , and the cost of those goods , requires either identifying the goods or using a convention to assume which goods were sold . This may be
referred to as a cost flow assumption or inventory identification assumption or convention . The following methods are available in many jurisdictions for
associating costs with goods sold and goods still on hand : Specific identification . Under this method , particular items are identified , and costs are
tracked with respect to each item . This may require considerable recordkeeping . This method can not be used where the goods or items are indistinguishable or
fungible . Average cost . The average cost method relies on average unit cost to calculate cost of units sold and ending inventory .
</item 2>

<item 3>
Cost of goods sold
If he deducted all the costs in 2008 , he would have a loss of $20 in 2008 and a profit of $180 in 2009 . The total is the same , but the timing is much
different . Most countries ' accounting and income tax rules ( if the country has an income tax ) require the use of inventories for all businesses that
regularly sell goods they have made or bought . Cost of goods for resale ( edit ) Cost of goods purchased for resale includes purchase price as well as all
other costs of acquisitions , excluding any discounts . Additional costs may include freight paid to acquire the goods , customs duties , sales or use taxes not
recoverable paid on materials used , and fees paid for acquisition . For financial reporting purposes such period costs as purchasing department , warehouse ,
and other operating expenses are usually not treated as part of inventory or cost of goods sold . For U.S. income tax purposes , some of these period costs must
be capitalized as part of inventory . Costs of selling , packing , and shipping goods to customers are treated as operating expenses related to the sale . Both
International and U.S.
</item 3>

<item 4>
Cost of goods sold
Costs are associated with particular goods using one of the several formulas , including specific identification , first - in first - out ( FIFO ) , or average
cost . Costs include all costs of purchase , costs of conversion and other costs that are incurred in bringing the inventories to their present location and
condition . Costs of goods made by the businesses include material , labor , and allocated overhead . The costs of those goods which are not yet sold are
deferred as costs of inventory until the inventory is sold or written down in value . Contents ( hide ) 1 Overview 2 Importance of inventories 3 Cost of goods
for resale 4 Cost of goods made by the business 5 Identification conventions 6 Example 7 Write - downs and allowances 8 Alternative views 9 Other terms 10 See
also 11 References 12 Further reading 12.1 Formal guidance Overview ( edit ) Many businesses sell goods that they have bought or produced . When the goods are
bought or produced , the costs associated with such goods are capitalized as part of inventory ( or stock ) of goods . These costs are treated as an expense in
the period the business recognizes income from sale of the goods .
</item 4>

<item 5>
Cost of goods sold
Determining costs requires keeping records of goods or materials purchased and any discounts on such purchase . In addition , if the goods are modified , the
business must determine the costs incurred in modifying the goods . Such modification costs include labor , supplies or additional material , supervision ,
quality control and use of equipment . Principles for determining costs may be easily stated , but application in practice is often difficult due to a variety
of considerations in the allocation of costs . Cost of goods sold may also reflect adjustments . Among the potential adjustments are decline in value of the
goods ( i.e. , lower market value than cost ) , obsolescence , damage , etc . When multiple goods are bought or made , it may be necessary to identify which
costs relate to which particular goods sold . This may be done using an identification convention , such as specific identification of the goods , first - in -
first - out ( FIFO ) , or average cost . Alternative systems may be used in some countries , such as last - in - first - out ( LIFO ) , gross profit method ,
retail method , or combinations of these .
</item 5>

<item 6>
Cost of goods sold
Cost of goods sold may be the same or different for accounting and tax purposes , depending on the rules of the particular jurisdiction . Certain expenses are
included in COGS . Expenses that are included in COGS can not be deducted again as a business expense . COGS expenses include : The cost of products or raw
materials , including freight or shipping charges ; The cost of storing products the business sells ; Direct labor costs for workers who produce the products ;
and Factory overhead expenses . Depreciation Importance of inventories ( edit ) Inventories have a significant effect on profits . A business that makes or buys
goods to sell must keep track of inventories of goods under all accounting and income tax rules . An example illustrates why . Fred buys auto parts and resells
them . In 2008 , Fred buys $100 worth of parts . He sells parts for $80 that he bought for $30 , and has $70 worth of parts left . In 2009 , he sells the
remainder of the parts for $180 . If he keeps track of inventory , his profit in 2008 is $50 , and his profit in 2009 is $110 , or $160 in total .
</item 6>

<item 7>
Cost of goods sold
The average cost method relies on average unit cost to calculate cost of units sold and ending inventory . Several variations on the calculation may be used ,
including weighted average and moving average . First - In First - Out ( FIFO ) assumes that the items purchased or produced first are sold first . Costs of
inventory per unit or item are determined at the time made or acquired . The oldest cost ( i.e. , the first in ) is then matched against revenue and assigned to
cost of goods sold . Last - In First - Out ( LIFO ) is the reverse of FIFO . Some systems permit determining the costs of goods at the time acquired or made ,
but assigning costs to goods sold under the assumption that the goods made or acquired last are sold first . Costs of specific goods acquired or made are added
to a pool of costs for the type of goods . Under this system , the business may maintain costs under FIFO but track an offset in the form of a LIFO reserve .
Such reserve ( an asset or contra - asset ) represents the difference in cost of inventory under the FIFO and LIFO assumptions . Such amount may be different
for financial reporting and tax purposes in the United States .
</item 7>

<item 8>
Cost of goods sold
Both International and U.S. accounting standards require that certain abnormal costs , such as those associated with idle capacity , must be treated as expenses
rather than part of inventory . Discounts that must be deducted from the costs of purchased inventory are the following : Trade discounts ( reduction in the
price of goods that a manufacturer or wholesaler provides to a retailer ) - includes a discount that is always allowed , regardless of the time of payment .
Manufacturer 's rebates -- is based on the dealer 's purchases during the year . Cash discounts ( a reduction in the invoice price that the seller provides if
the dealer pays immediately or within a specified time ) -- it may reduce COGS , or it may be treated separately as gross income . Value added tax is generally
not treated as part of cost of goods sold if it may be used as an input credit or otherwise recoverable from the taxing authority . Cost of goods made by the
business ( edit ) The cost of goods produced in the business should include all costs of production . The key components of cost generally include : Parts , raw
materials and supplies used , Labor , including associated costs such as payroll taxes and benefits , and Overhead of the business allocable to production .
</item 8>

<item 9>
Cost of goods sold
Cost of goods sold - wikipedia Cost of goods sold Jump to : navigation , search Accounting Historical cost Constant purchasing power Management Tax Major types
( show ) Auditing Cost Forensic Financial Fund Governmental Management Social Tax Key concepts ( show ) Accounting period Accrual Constant purchasing power
Economic entity Fair value Going concern Historical cost Matching principle Materiality Revenue recognition Unit of account Selected accounts ( show ) Assets
Cash Cost of goods sold Depreciation / Amortization Equity Expenses Goodwill Liabilities Profit Revenue Accounting standards ( show ) Generally - accepted
principles Generally - accepted auditing standards Convergence International Financial Reporting Standards International Standards on Auditing Management
Accounting Principles Financial statements ( show ) Annual report Balance sheet Cash - flow Equity Income Management discussion Notes to the financial
statements Bookkeeping ( show ) Bank reconciliation Debits and credits Double - entry system FIFO and LIFO Journal Ledger / General ledger T accounts Trial
balance Auditing ( show ) Financial Internal Firms Report People and organizations ( show ) Accountants Accounting organizations Luca Pacioli Development ( show
) History Research Positive accounting Sarbanes -- Oxley Act Business portal Cost of goods sold ( COGS ) refers to the carrying value of goods sold during a
particular period .
</item 9>

<item 10>
Cost of goods sold
U.S. Internal Revenue Service Publication 334 , Tax Guide for Small Business , pages 27 -- 29 . U.S. Financial Accounting Standards Board ASC 330 . ( hide )
Accounting Type Financial accounting Cost accounting Statements Balance sheet Income statement Cash flow statement Terms Revenue Cost of goods sold Operating
expenses Capital expenditure Depreciation Gross profit Net profit Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Cost_of_goods_sold&oldid=834402233 '' Categories : Inventory Accounting terminology Costs Sales Hidden categories :
All articles with dead external links Articles with dead external links from April 2018 Webarchive template wayback links Talk Contents About Wikipedia Deutsch
فارسی Bahasa Indonesia עברית Lietuvių 日本 語 Polski Русский Українська 中文 3 more Edit links This page was last edited on 5 April 2018 , at 14 : 57 . Text is
available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and
Privacy Policy .
</item 10>
=== END ITEM id=37ab120d41e2c9ad ===

=== ITEM id=e21e3dfc2d00fc0b ===
Gold sentences and facts:
Sentence 0: A transfer case is a part of the drivetrain of four - wheel - drive , all - wheel - drive , and other multiple powered axle vehicles . The transfer case transfers power from the transmission to the front and rear axles by means of drive shafts . It also synchronizes the difference between the rotation of the front and rear wheels , and may contain one or more sets of low range gears for off - road use .
  Fact 0.0: A transfer case is a part of the drivetrain of four-wheel-drive, all-wheel-drive, and other multiple powered axle vehicles.
  Fact 0.1: A transfer case transfers power from the transmission to the front and rear axles by means of drive shafts.
  Fact 0.2: A transfer case synchronizes the difference between the rotation of the front and rear wheels.
  Fact 0.3: A transfer case may contain one or more sets of low range gears for off-road use.

Retrieved items:
<item 1>
Transfer case
Transfer case - wikipedia Transfer case Jump to : navigation , search For the US military use of the term , see coffin . Inside of a 231 New Process Gear
transfer case . Part - time / Manual , shift on the fly A transfer case is a part of the drivetrain of four - wheel - drive , all - wheel - drive , and other
multiple powered axle vehicles . The transfer case transfers power from the transmission to the front and rear axles by means of drive shafts . It also
synchronizes the difference between the rotation of the front and rear wheels , and may contain one or more sets of low range gears for off - road use .
Contents ( hide ) 1 Functions 2 Types 2.1 Drive type 2.1. 1 Gear - driven 2.1. 2 Chain - driven 2.2 Housing type 2.2. 1 Married 2.2. 2 Divorced / independent
2.3 Transfer case shift type 2.3. 1 M.S.O.F. 2.3. 2 E.S.O.F.
</item 1>

<item 2>
Transfer case
See also ( edit ) NP - 205 AMC / Jeep Transmissions Jeep four wheel drive systems Mitsubishi Super Select References ( edit ) Jump up ^
http://www.cdxetextbook.com/trans/finalDrives/allWheel/transfercasediff.html Transfer case differential action Jump up ^
http://www.fourwheeler.com/techarticles/68958_how_transfer_cases_work/index.html How Transfer Cases Work Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Transfer_case&oldid=805616544 '' Categories : Vehicle technology Four - wheel drive layout Hidden categories : All
articles with unsourced statements Articles with unsourced statements from May 2015 Articles with unsourced statements from March 2012 Talk Contents About
Wikipedia فارسی 日本 語 Polski Русский Simple English Svenska Українська Edit links This page was last edited on 16 October 2017 , at 15 : 12 . Text is available
under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy
.
</item 2>

<item 3>
Transfer case
By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit
organization . About Wikipedia
</item 3>

<item 4>
Transfer case
This is necessary because the front and rear tires never turn at the same speed . Different rates of tire rotation are generally due to different tire diameters
( since front and rear tires inevitably wear at different rates ) and different gear ratios in the front and rear differentials since manufacturers will often
have a slightly lower ratio in the front vs. the rear to help with control ( such as a 3.55 : 1 in the rear differential and a 3.54 : 1 in the front
differential ) . If the transfer case did not make up the difference between the two different rates of rotation , binding would occur and the transfer case
could become damaged . This is also why a transfer case that is not designed for on - road use must never be driven on dry pavement . Transfer cases designed
for off - road use can mechanically lock the front and rear axles when needed ( e.g. when one of the axles is on a slippery surfaces or stuck in mud , whereas
the other has better traction ) . This is the equivalent to the differential lock . The transfer case may contain one or more sets of low range gears for off -
road use .
</item 4>

<item 5>
Transfer case
Married transfer cases are bolted directly to the transmission , usually between the transmission 's output shaft and the rear or main driveshaft . Sometimes a
married transfer case is an integral part of the transmission and the two components share the same housing or `` case '' , as is commonly found on recent
Subaru products and some other all - wheel - drive cars . Divorced / independent ( edit ) A divorced or independent transfer case is completely separate from
the transmission . It is located further down the driveline than a married transfer case and connected to the transmission output shaft by a short driveshaft .
Independent transfer cases are used on very long wheelbase vehicles , such as commercial trucks or military trucks . Transfer case Shift type ( edit ) M.S.O.F.
( edit ) Manual Shift On - the - Fly transfer cases have a selector lever on the driver 's side floor transmission hump and may also have either two sealed
automatic front axle locking hubs or two manual front axle hub selectors of `` LOCK '' and `` UNLOCK '' or `` FREE '' . To engage the four - wheel - drive
system the vehicle must be moving at a low speed , the speed at which 4x4 can be engaged depends on the vehicle .
</item 5>

<item 6>
Transfer case
The transfer case may contain one or more sets of low range gears for off - road use . Low range gears are engaged with a shifter or electronic switch . On many
transfer cases , this shifter is the same as the one that selects 2WD or 4WD operation . Low range gears allow the vehicle to drive at much slower speeds while
still operating within the usable power band / RPM range of the engine . This also increases the torque available at the axles . Low - range gears are used
during slow - speed or extreme off - road maneuvers , such as rockcrawling , navigating dangerous roads , or when pulling a heavy load . This feature is often
absent on all - wheel - drive cars . Some very large vehicles , such as heavy equipment or military trucks , have more than one low - range gear . Types ( edit
) Transfer cases used on `` part - time '' four - wheel - drive off - road vehicles such as trucks , truggies , rock - crawling vehicles , and some military
vehicles generally allow the driver to select 2WD or 4WD , as well as high or low gear ranges .
</item 6>

<item 7>
Transfer case
1 M.S.O.F. 2.3. 2 E.S.O.F. 3 See also 4 References Functions ( edit ) The transfer case receives power from the transmission and sends it to both the front and
rear axles . This can be done with gears , hydraulics , or chain drive . On some vehicles , such as four - wheel - drive trucks or vehicles intended for off -
road use , this feature is controlled by the driver . The driver can put the transfer case into either `` two - wheel - drive '' or `` four - wheel - drive ''
mode . This is sometimes accomplished by means of a shifter , similar to that in a manual transmission . On some vehicles this may be electronically operated by
a switch instead . Some vehicles , such as all - wheel - drive sports cars , have transfer cases that are not selectable . Such a transfer case is permanently
`` locked '' into all - wheel - drive mode . Transfer cases that are designed to allow for normal road use synchronize the difference between the rotation of
the front and rear wheels , in much the same way the differential acts on a given axle . This is necessary because the front and rear tires never turn at the
same speed .
</item 7>

<item 8>
Transfer case
This is only for the four - wheel - drive high setting . To engage the four - wheel - drive low setting , the vehicle must be stopped and the transmission must
be shifted to neutral , then the four - wheel - drive low can be selected . ESOF ( edit ) Electronic Shift On - the - Fly ( ESOF ) transfer cases have a dash -
mounted selector switch or buttons with front sealed automatic locking axle hubs or drive flanges . Some vehicles , such as Jeeps , also have what is called
SelecTrac , which has a slider switch on the center console . Unlike the manual transfer case , this system has a transfer case motor . To engage the four -
wheel - drive system the vehicle must be moving at a lower speeds , the speed at which 4x4 can be engaged depends on the vehicle . This is only for the four -
wheel - drive high setting . To engage the four - wheel - drive low setting , the vehicle must be stopped and the transmission must be shifted to neutral , then
the four - wheel - drive low can be selected .
</item 8>

<item 9>
Jeep
Off - road abilities Jeep Wrangler off - roading . Jeep advertising has always emphasized the brand 's vehicles ' off - road capabilities . Today , the Wrangler
is one of the few remaining four - wheel - drive vehicles with solid front and rear axles . These axles are known for their durability , strength , and
articulation . New Wranglers come with a Dana 44 rear differential and a Dana 30 front differential . The upgraded Rubicon model of the JK Wrangler is equipped
with electronically activated locking differentials , Dana 44 axles front and rear with 4.10 gears , a 4 : 1 transfer case , electronic sway bar disconnect and
heavy duty suspension . Another benefit of solid axle vehicles is they tend to be easier and cheaper to `` lift '' with aftermarket suspension systems . This
increases the distance between the axle and chassis of the vehicle . By increasing this distance , larger tires can be installed , which will increase the
ground clearance , allowing it to traverse even larger and more difficult obstacles . In addition to higher ground clearance , many owners aim to increase
suspension articulation or `` flex '' to give their Jeeps greatly improved off - road capabilities .
</item 9>

<item 10>
Transfer case
Those used in sports cars and performance sedans are usually `` transparent '' to the driver ; there is no shifter or select lever . Drive type ( edit ) Gear -
driven ( edit ) There are two different types of internal power - transfer mechanism found in most transfer cases . Gear - driven transfer cases use sets of
gears to drive either the front or both the front and rear driveshafts . These are generally strong , heavy units that are used in large trucks , but there are
currently several gear drive cases in production for passenger cars . Chain - driven ( edit ) Chain - driven transfer cases use a chain to drive most often only
one axle , but can drive both axles . Chain - driven transfer cases are quieter and lighter than gear - driven ones . They are used in vehicles such as compact
trucks , full size trucks , Jeeps and SUVs . Some off - road driving enthusiasts modify their vehicles to use gear - driven transfer cases , accepting the
additional weight and noise to gain the extra strength they generally provide . Housing type ( edit ) Married ( edit ) Transfer cases are also classified as
either `` divorced '' / independent or `` married '' .
</item 10>
=== END ITEM id=e21e3dfc2d00fc0b ===
