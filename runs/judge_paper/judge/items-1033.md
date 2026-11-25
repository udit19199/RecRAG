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

This file has 4 items.

=== ITEM id=99c62fb98b649851 ===
Gold sentences and facts:
Sentence 0: One of the major sources of Gen Z’s distress, of course, is climate change.
  Fact 0.0: Climate change is one of the major sources of Gen Z's distress.
Sentence 1: In other words, first, we need to improve our understanding of drivers of sea surface temperature patterns—decadal coupled variability; the pace and spatial structure of ocean heat uptake; and forcing by aerosols, greenhouse gases, and volcanoes—as well as the relative timing of these drivers.
  Fact 1.0: We need to improve our understanding of the drivers of sea surface temperature patterns.
  Fact 1.1: Improving the understanding of drivers of sea surface temperature patterns is the first step.
  Fact 1.2: One driver of sea surface temperature patterns is decadal coupled variability.
  Fact 1.3: Another driver is the pace of ocean heat uptake.
  Fact 1.4: Another driver is the spatial structure of ocean heat uptake.
  Fact 1.5: Another driver is forcing by aerosols.
  Fact 1.6: Another driver is forcing by greenhouse gases.
  Fact 1.7: Another driver is forcing by volcanoes.
  Fact 1.8: We need to improve our understanding of the relative timing of these drivers.

Retrieved items:
<item 1>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
Climate change isn't going away, and it isn't going to get any better — at least if we keep legislating as we have been. In Democracy in a Hotter Time: Climate
Change and Democratic Transformation, a multidisciplinary collection of subject matter experts discuss the increasingly intertwined fates of American ecology
and democracy, arguing that only by strengthening our existing institutions will we be able to weather the oncoming "long emergency."

In the excerpt below, contributing author and Assistant Professor of Environment and Sustainability at the University at Buffalo, Holly Jean Buck, explores how
accelerating climate change, the modern internet and authoritarianism's recent renaissance are influencing and amplifying one another's negative impacts, to the
detriment of us all.

MIT Press

Excerpted from Democracy in a Hotter Time: Climate Change and Democratic Transformation, edited by David W. Orr. Published by MIT Press. Copyright © 2023. All
rights reserved.
</item 1>

<item 2>
Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' (Fortune,
2023-10-10T20:17:09+00:00)
6, 2021, following the election defeat of former President Donald Trump. Then there are school shootings, which have only increased in frequency since Columbine
in 1999, with more students documenting the terror on their phones and sharing it online. One of them was the Parkland shooting of 2018, which tragically
created the first spokespeople of the post-millennial generation.

Gen Z’s mental health

Gen Z has the worst reported mental health of any generation—45% of young people report having “excellent” or “very good” mental health, according to a 2018
report by the American Psychological Association.

One of the major sources of Gen Z’s distress, of course, is climate change. Nearly seven in 10 Gen Zers say they experience anxiety when viewing climate change
content on social media, according to a 2021 Pew Research report.
</item 2>

<item 3>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
Neighbors drop by without notice.

“We’ve created what existed 100 years ago,” Ethan Hughes said.

On a steamy August morning, Kalmus found himself huddled around a faded picnic table at the center of the homestead in Belfast, Maine, sipping his coffee with a
rare breed — people as alarmed by climate change as him.

climate scientist environmental (Evan Bush / NBC News)

Ethan, the charismatic, folksy co-founder of the Possibility Alliance, started the day off with a poem about a deer in the forest and then began to outline
daily chores — chopping wood, building a new deck, caring for goats.

Soon, talk turned to climate change and Kalmus’ grief began to flow as he shared findings from unpublished research on his projections of heat in moderate
emissions scenarios through the year 2300.

“Even in the coolest time of day, people will be dying if they don’t have air conditioning,” Kalmus told the group.

“Everything is already spinning out of control,” said Sarah Wilcox-Hughes, dismayed.
</item 3>

<item 4>
Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' (Fortune,
2023-10-10T20:17:09+00:00)
The “disturbing images” warning is one that Gen Z, the first fully digital native cohort, is used to encountering on their lingua franca: social media posts. On
Saturday, the Islamic militant group Hamas launched an unprecedented attack on Israel, resulting in the worst war the region has seen in 50 years. In three
days, more than 1,500 people have died on both sides, and graphic, difficult-to-watch videos have blanketed the internet—from Israeli civilians being captured,
tortured, and killed by Hamas militants, to Palestinian civilians screaming in grief, and people on both sides attending to their dead and injured among the
rubble.

If Gen Z feels like they’ve seen it all, in some ways, they have. This generation, ages 11 to 26, has already lived through numerous historic events, ranging
from a once-per-century pandemic, to the Jan. 6 insurrection—an event unseen for centuries in American politics.
</item 4>

<item 5>
Consistent policy is key to unlocking the climate tech boom, but the UK’s U-turn isn’t helping (TechCrunch, 2023-10-13T17:00:20+00:00)
because it offers the best possible place to make their vision happen.” Unless his long-term vision was only meant to last until the end of the summer, for
entrepreneurs, innovators and businesses to thrive and unlock the economic potential that comes from creating new industries, we need a consistent approach from
the government.

We have enormous fiscal potential. Within the U.K. climate tech community, we are working to create high-paying jobs and value for investors across just about
every asset class. And collectively — hell, individually — our solutions could genuinely change the world.

Of course, this is the common cause we should be united around. Our company is trying to move industry away from a reliance on fossil fuels to make chemicals,
among other things. But the impact of these political spasms and contradictions in our climate commitments has on, say, an EV battery business — which has just
seen the market demand for its output slide down the road by five years — isn’t hard to appreciate.
</item 5>

<item 6>
The shipping industry’s boom-and-bust cycle is so severe carriers face going from bumper profits to losing money in the space of a few years: 'I'm certainly
concerned' (Fortune, 2023-11-12T17:51:37+00:00)
Consumers gearing up to buy the latest imported appliances, clothes or electronic gadgets this holiday season might want to spare a thought for the companies
that will struggle to make money for the next few years hauling products across the ocean.

That’s because the container shipping industry, cast as the Grinch that spoiled Christmas over the past two years with record-high freight rates and slow
deliveries, is returning to its pre-pandemic place in the corporate world: perennial underachiever Charlie Brown.

The biggest carriers posted net income totaling $364 billion in 2021 and 2022, according to figures compiled by industry veteran John McCown, after a decade of
scant profits. They’ll likely drift back into the red this quarter as the rates they charge fall below costs and look to stay there for the foreseeable future.
</item 6>

<item 7>
The shipping industry’s boom-and-bust cycle is so severe carriers face going from bumper profits to losing money in the space of a few years: 'I'm certainly
concerned' (Fortune, 2023-11-12T17:51:37+00:00)
Companies including clothing brand Under Armour Inc. and furniture maker Lovesac Co. cited relief from lower ocean shipping expenses over the past quarter.

With inflation eating away at their paychecks, consumers are being cautious about spending and seeking cheaper ways to have parcels delivered.

“Our own data tells us that the wider economic picture may be having an influence on the services our customers opt for, with many looking for more
cost-effective shipping options,” said Karen Reddington, president of FedEx Express Europe. “We expect external business conditions to be challenging in the
near term, and there remains uncertainty with respect to the timing of demand recovery.”

Read More: Powell Says Fed to Be Careful, Won’t Hesitate to Hike If Needed

For the container carriers, the cost of moving merchandise can’t stay this low indefinitely, because their expenses are heading in the opposite direction.
</item 7>

<item 8>
La desigualdad del estrés por calor (Eos: Earth And Space Science News, 2023-09-26T12:15:35+00:00)
“No solo necesitamos entender mejor dónde hace calor, sino también cuáles son los factores socioeconómicos para los habitantes de esas zonas”, dijo Whitely
Binder. “Podemos comenzar a analizar lo que hay en las islas de calor y utilizarlo para comenzar a informar las decisiones políticas que vamos a tomar”.

—Rebecca Owen (@beccapox), Escritora de ciencia

This translation by Nel Rodriguez Sepulveda was made possible by a partnership with Planeteando. Esta traducción fue posible gracias a una asociación con
Planeteando.

This news article is included in our ENGAGE resource for educators seeking science news for their classroom lessons. Browse all ENGAGE articles , and share with
your fellow educators how you integrated the article into an activity in the comments section below.

Text © 2023. The authors. CC BY-NC-ND 3.0

Except where otherwise noted, images are subject to copyright.
</item 8>

<item 9>
Walks, tech and protein: how to parent your own parents (The Guardian, 2023-10-10T09:00:35+00:00)
Parenting no longer starts and stops with our children. Nor is it confined to those who have children. In a time of unrelenting change and ever-extending life,
most of us will – at some stage – find ourselves “parenting” our own parents.

Indeed, many of us – particularly those who had families later – will find ourselves simultaneously parenting our kids and our parents. In one breath we’ll be
begging our children to swap French fries for vegetables, and in the next breath we’ll be urging our parents to exchange cake for sardines. Little wonder
today’s midlifers are known as the sandwich generation.

The relentless pace of technology – accelerated by the Covid-19 pandemic – has made everything infinitely more complicated, leaving many older (and not-so-old)
people baffled and frustrated.
</item 9>

<item 10>
La desigualdad del estrés por calor (Eos: Earth And Space Science News, 2023-09-26T12:15:35+00:00)
La correlación de los índices de calor mapeados con los datos del censo reveló que los vecindarios de bajos ingresos y los residentes de color experimentaron
temperaturas más altas y mayor humedad, lo que en conjunto amplifica el estrés por calor. Las zonas censales con mayores ingresos sufrieron menos estrés por
calor. El estrés por calor también fue generalmente mayor en las zonas con un mayor porcentaje de residentes negros y afroamericanos.

Áreas censales de Baltimore coloreadas según el índice de calor y el porcentaje de residentes negros. Una comparación indica que donde una mayor proporción de
residentes son negros, el estrés por calor es mayor.
</item 10>

<item 11>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
Roosters crowed. A tear dripped down Ethan’s cheek.

“Let’s dedicate our work today to the Amazon,” Ethan said, outlining for the group how the work planned would sequester carbon and show people they could live
without fossil fuels and that they could take meaningful action against climate change, even as world governments failed them.

(Estimates of when the Amazon could “tip” and transition to savannah vary. Timothy Lenton, a University of Exeter professor who studies climate tipping points
and the Amazon’s resilience, said most estimates suggest the Amazon could cross a threshold as soon as 2 degrees C of warming.)

Life at the Possibility Alliance carried on. Will Foley, who hitchhiked from Florida to stay at the homestead, swung a scythe to clear brush. Frankie Williams
and Max Kurke, recent college graduates living at the alliance as summer apprentices, piled wood while wearing oversize sun hats. Ethan corralled goats for “mob
grazing,” a process he says helps sequester carbon.
</item 11>

<item 12>
Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' (Fortune,
2023-10-10T20:17:09+00:00)
But they’re not just reading about the detrimental effects of human-caused global warming, they’re living through the consequences themselves.

This summer reached record-breaking temperatures, with July being the hottest month the planet has seen in over 100,000 years. As a result, Arizona experienced
a monthlong heat wave with temperatures at or above 110 degrees every day. Deadly fires broke out across the Mediterranean; suffocating smoke from Canadian
wildfires blanketed New York City and the Northeast for days; and ice melt in the Arctic accelerated.

And extreme heat is likely here to stay—and get worse—unless countries can rapidly reduce their carbon emissions. That’s why Gen Z is more concerned with
sustainability than any generation before them.
</item 12>

<item 13>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
What comes to mind? If you recombine the familiar characters from these stories, perhaps it looks like climate activists using the capabilities of the Internet
to further both networked protest and energy democracy. In particular, advocacy for a version of “energy democracy” that looks like wind, water, and solar;
decentralized systems; and local community control of energy.

In this essay, I would like to suggest that this is not actually where the three forces of rising authoritarianism x climate change x tech platforms domination
leads. Rather, the political economy of online media has boxed us into a social landscape wherein both the political consensus and the infrastructure we need
for the energy transition is impossible to build. The current configuration of the Internet is a key obstacle to climate action.

The possibilities of climate action exist within a media ecosystem that has monetized our attention and that profits from our hate and division. Algorithms that
reap advertising profits from maximizing time-on-site have figured out that what keeps us clicking is anger.
</item 13>

<item 14>
La desigualdad del estrés por calor (Eos: Earth And Space Science News, 2023-09-26T12:15:35+00:00)
Mapeando la disparidad de calor

Estudios previos sobre el calor urbano han utilizado datos satelitales para estimar temperaturas de la superficie terrestre. En cambio, Chackraborty y sus
colegas evaluaron el estrés por calor utilizando el índice de calor del Servicio Nacional Meteorológico de los E.E. U.U. y el índice de humedad del Servicio
Nacional Meteorológico de Canadá (humidex), combinando la temperatura del aire y la humedad para describir mejor cómo se siente el calor. Utilizar modelos que
combinan estas variables fue una manera más certera de catalogar el estrés por calor de las ciudades estadounidenses entre 2014 y 2018.
</item 14>

<item 15>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
Months later, after returning “kicking and screaming” to the “quote-unquote real world,” Kalmus was planning a return trip to the Possibility Alliance over the
winter holidays.

He was still mulling why the Possibility Alliance’s “low energy” lifestyle had felt so meaningful and what was missing from modern life.

Ethan Hughes his goats out to

“Everything is too fast,” Kalmus said. “It’s obvious to a lot of people, but it’s a new revelation to me. There’s nothing more important in life than
relationships.”

This article was originally published on NBCNews.com
</item 15>

<item 16>
La desigualdad del estrés por calor (Eos: Earth And Space Science News, 2023-09-26T12:15:35+00:00)
En la década de 1930, el gobierno de los Estados Unidos calificó a los vecindarios según su capacidad de inversión. Muchos de los vecindarios que albergaban
poblaciones pobres y minoritarias, especialmente de residentes negros y afroamericanos, fueron considerados inversiones mas riesgosas y, por lo tanto,
recibieron menos fondos para desarrollo y programas de vivienda. Hoy día, estos vecindarios marginalizados sufren condiciones ambientales mucho peores que otras
partes de sus respectivas ciudades. Tienen menos cubierta de árboles y experimentan temperaturas superficiales mayores que vecindarios que originalmente no
fueron marginalizados.

Las disparidades eran penetrantes, dijo Chakraborty. Los vecindarios marginalizados, generalmente calificados con una “D” por prestamistas, tenían un índice de
calor mayor comparado con aquellos que eran calificados con una “A”.
</item 16>

<item 17>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
Most important, models do not re-create observed cooling in the equatorial and subtropical eastern Pacific and Southern Ocean (Figure 2, right), and it is
concerning that all models have the same sign of error in the trend patterns [e.g., Wills et al., 2022]. In addition, models strongly disagree about the
timescales over which future warming may occur in these regions.

The advent of global coupled storm- and cloud-resolving models, which currently can simulate only a few months to years at a time, presents an exciting
opportunity.

Our limited knowledge of the drivers of surface warming pattern biases in climate models hampers our ability to evaluate these biases. We do not know whether
these models’ inability to reproduce observed trends is due to biases in how they represent the spectrum of internal variability, aerosol forcing,
ocean-atmosphere interactions, deep and shallow atmospheric convection, Southern Ocean cloud-radiative properties, ocean thermocline depth, and ocean
circulation or something else entirely.
</item 17>

<item 18>
La desigualdad del estrés por calor (Eos: Earth And Space Science News, 2023-09-26T12:15:35+00:00)
“Encontramos que la disparidad actual era bastante consistente entre las ciudades: en más del 90% de las ciudades que consideramos, encontramos desigualdades a
la exposición al calor relacionadas tanto en los ingresos como en la raza”, dijo TC Chakraborty, científico de la Tierra en el Laboratorio Nacional del Noroeste
Pacífico y autor principal del estudio. Este tipo de información puede ayudar a los líderes de ciudad a usar mejores medidas de las disparidades del calor, así
como proteger a las poblaciones en mayor riesgo de exposición al calor.
</item 18>

<item 19>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
The three strands of research have converged over the past few years, highlighting that understanding the pattern effect benefits from—and perhaps
requires—contributions from virtually all climate research communities studying large-scale ocean-atmosphere coupling and the dynamics that set regional to
global responses to external forcing.

To foster this work across communities, 140 scientists from around the world and from different disciplines gathered for a US CLIVAR (U.S. Climate Variability
and Predictability Program) workshop in May 2022 in Boulder, Colo. Below, we present the current consensus that emerged during the workshop and raise questions
that require urgent attention from scientists.

The Past Is a Poor Analogue for the Future

Research on the pattern effect has exposed limitations of using equilibrium climate sensitivity to constrain future warming and, conversely, of using recently
observed warming to constrain equilibrium climate sensitivity. The foremost implication of the pattern effect is that historically observed climate change does
not constrain the upper limit of climate sensitivity [Sherwood et al., 2020; Forster et al., 2021].
</item 19>

<item 20>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
To restore the energy balance, the planet must warm. But warming causes changes in the climate system: The concentration of water vapor—a greenhouse gas—in the
atmosphere increases, the spatial coverage of highly reflective snow and sea ice decreases, and cloud properties change. These and other radiative feedbacks
amplify or dampen how much the planet warms in response to the forcing. Hence, for a given forcing, the feedbacks determine the climate sensitivity.

A degree of global warming spread out evenly will cause a different radiative response than if that same warming is concentrated in a climate sweet spot.

For decades, researchers assumed that global mean radiative feedbacks mostly depend on global mean temperature [Gregory et al., 2004].
</item 20>

<item 21>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
A number of works outline the contours and dynamics of the current media ecology and what it does — Siva Vaidhyanathan’s Antisocial Media, Safiya U. Noble’s
Algorithms of Oppression, Geert Lovink’s Sad by Design, Shoshana Zuboff’s Surveillance Capitalism, Richard Seymour’s The Twittering Machine, Tim Hwang’s
Subprime Attention Crisis, Tressie McMillan Cottom’s writing on how to understand the social relations of Internet technologies through racial capitalism, and
many more. At the same time, there’s reasonable counter-discussion about how many of our problems can really be laid at the feet of social media. The research
on the impacts of social media on political dysfunction, mental health, and society writ large does not paint a neat portrait.
</item 21>

<item 22>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
And most pressing for the communities relying on the coupled models’ climate change projections, To what extent are the remaining carbon budgets [Zhou et al.,
2021] with respect to global warming targets, rates of near-term future warming, and detection and attribution efforts contingent on the pattern effect?

A Collaborative Outlook

Working to solve the problems outlined above at the intersection of different disciplines and with input from different research communities will likely benefit
related research into, for example, the sensitivity of tropical cyclones and future rainfall changes in the southwestern United States or in South America to
surface warming patterns.

Promising new tools and observations are beginning to emerge. They include targeted model experiments and intercomparisons, extended global satellite
observations of clouds and radiation, Argo floats sampling deep-ocean heat uptake, estimates of radiative imbalances before the year 2000, and new observational
constraints on clouds’ sensitivity to environmental controlling factors. These developments encourage an optimistic outlook on our ability to quantify the
pattern effect and its implications over the next couple of years.
</item 22>

<item 23>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
Twice, he’s been arrested during climate protests.

His social media posts often evoke heartbreak.

“We’re in the process of losing basically everything,” he wrote recently on X.

Is anyone listening?

climate scientist natural farming (Evan Bush / NBC News)

Not his suburban neighbors in North Carolina, who continue to drive emissions-belching SUVs to big-box stores. Perhaps not even some in the science community,
like those who kicked him out of the American Geophysical Union’s annual fall meeting last year after he unfurled a banner and protested on the major science
conference’s biggest stage.

He has grown increasingly frustrated with President Joe Biden, who signed the Inflation Reduction Act as his signature climate bill. Kalmus thinks it does too
little to shut out the fossil fuel industry.

(Rhodium Group, a nonpartisan think tank, estimates greenhouse gas emissions will drop 32%-42% below 2005 levels by 2030, well short of Biden’s own benchmark
for progress. The inflation bill is responsible for about a quarter of that projected decrease.)
</item 23>

<item 24>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
BELFAST, Maine — How does it feel to be a climate scientist in 2023?

Let’s set the scene. Hundreds of picnickers are sprawled across a lawn speckled with colorful blankets. Yachts bob in the bay. Guitars blare, children wrestle
and dancers twirl across a concert lawn in the golden glow of a dying summer night.

It’s a picture-perfect August evening for the seaside town of Belfast’s “Summer Nights” concert series.

Off to the side, Peter Kalmus sits alone. His eyes are closed. His toes are tucked beneath his body. He’s catching an occasional side-eye from onlookers.

“When I’m meditating, I don’t feel anxious,” Kalmus later says, explaining that he strives for two hours of meditation a day to keep his climate anxiety at bay.
Otherwise, “it’s completely overwhelming.”

Many of us avoid thinking too deeply about the worst scenarios of climate change. He is fixated on them.
</item 24>

<item 25>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
He is fixated on them.

Kalmus, 49, thinks civilization is on the path to break down, the Biden administration is clueless on climate, and that he might get fired from his job at NASA
if he is arrested for a third time protesting what he views as downright madness: the continued use of fossil fuels.

Kalmus visited the seaside town of Belfast, Maine, in July and August to visit the Possibility Alliance homestead as he looked for inspiration on how to best
live with the effects climate change. (Evan Bush / NBC News)

“It’s kind of weird and lonely,” he said. “People don’t want to talk about this stuff at parties.”

Kalmus, whose strident tweets reach some 340,000 followers on X, is at the vanguard of a group of scientists who are increasingly disillusioned and growing
convinced that more radical action is needed to wake up political leaders.
</item 25>

<item 26>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
One in 10 reported feeling symptoms of anxiety or depression over climate change.

No place is safe

To Kalmus, the best answer he’s found has been to channel his frustrations into protest and civil disobedience, finding community and purpose at the Mountain
Valley Pipeline protests and on the streets of New York City during its Climate Week. Outside of meditation, speaking his mind at these events has been the best
antidote to his climate despair.

When he earned his first arrest, by chaining himself to a bank in Los Angeles last year, he finally felt acknowledged.

climate scientist arrest (Courtesy Brian Emerson/Climate Ad Project)

“It was a good combination of risk and sort of communicating emotional truth. So I was able to dig really deep, and say exactly how I felt in the moment,” he
said.

His outburst went viral. His employer, NASA, sent him a letter expressing concern over his two arrests, he says.

(“I’m speaking on my own behalf and that has to be super clear,” he said in an interview. “Not as a NASA climate scientist.
</item 26>

<item 27>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
The inflation bill is responsible for about a quarter of that projected decrease.)

“He brags about how he thinks we should consider him a climate champion because he reentered the Paris accord. That Paris Agreement will take us to about 3
degrees Celsius of global heating,” Kalmus said. “I don’t think we’ll have a civilization at 3 degrees Celsius.”

(A 2022 United Nations report estimated global temperatures would rise between 2.1 and 2.9 degrees Celsius by 2100 if countries held to their climate
commitments. Many countries remain off that pace.)

Climate change is weighing on scientists, but also everyday Americans.

A 2022 poll found almost two-thirds of Americans say they have been affected by extreme weather they believe was at least partially due to climate change.

About 27% of Americans say they are “very worried” about climate change; another 27% just avoid the subject as best they can. One in 10 reported feeling
symptoms of anxiety or depression over climate change.
</item 27>

<item 28>
Consistent policy is key to unlocking the climate tech boom, but the UK’s U-turn isn’t helping (TechCrunch, 2023-10-13T17:00:20+00:00)
People will lose jobs. And action, alas, will be further delayed.

In the end, it’s simple: Inconsistency breeds uncertainty. So give us consistency. It is possible for our sector to succeed in spite of bad policies. But to
thrive, we need consistent resolutions and behavior from a government that actually cares about leading in this critical fight. Given the colossal stakes, which
were at least recently taken seriously, is that really asking too much?
</item 28>

<item 29>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
Early in the 2010s, a substantial discrepancy was noted between estimates of climate sensitivity derived from climate models and estimates based on the observed
warming record and radiative balance, the balance between incoming and reflected solar radiation and outgoing terrestrial radiation. Estimates based on observed
warming pointed to much lower values than those derived from models. A key breakthrough toward solving this conundrum has been the recognition of the pattern
effect, the process whereby climate sensitivity depends on the geographic pattern of surface warming. This advance was rated as one of the most promising
avenues for further constraining climate sensitivity in the future [Forster et al., 2021].

Forcing, Feedbacks, and Climate Sweet Spots

Adding greenhouse gases to Earth’s atmosphere leads to a global energy surplus (less terrestrial radiation escapes to space), referred to as forcing. To restore
the energy balance, the planet must warm.
</item 29>

<item 30>
'They seemed primed to take over': How the Great Dying doomed the 'beast tooth' and set the stage for the dawn of the dinosaurs (Live Science: The Most
Interesting Articles, 2023-10-01T08:00:25+00:00)
Some researchers even argue for a possible "bottleneck" effect, where the few life-forms that survived the Neoproterozoic ice ages (Snowball Earth or not) were
able to rapidly fill emerging niches as Earth thawed.

Related: 'Once again, innovation and proliferation ended with catastrophe': The environmental disaster of plants taking over the world

A major glacial event occurred at the end of the following period of the Paleozoic, the Ordovician, around 450 million years ago, as chemical weathering
outpaced the volcanic emissions of gas and atmospheric CO2 levels dropped. The resulting cooling caused a buildup in ice mass on the large South Pole–centered
supercontinent of Gondwana. Sea levels dropped. Much of the coastal habitat that had been home to primitive mollusks and crustaceans disappeared. Some of the
creatures scraped by, but about half of all existing genuses perished.
</item 30>

<item 31>
New ultrathin materials can pull climate-warming CO2 from the air (Science News For Students, 2023-12-15T11:30:00+00:00)
These materials are harder to make than MXenes. So most research on them has been done with math or computer models. But MBenes could have unique mechanical and
electrical traits, Ozkan’s team finds. And those just might help them last through many, many cycles of CO 2 capture and release.

Each year, human activities spew tens of billions of tons of CO 2 into the air. That’s equivalent to the weight of tens of billions of elephants. Carbon
absorbers, including MXenes or MBenes, can’t catch it all. As such, they’re never going to replace the need for reducing emissions of CO 2 and other greenhouse
gases.

Still, given the urgency of climate change, even a quick or partial fix — such as carbon capture — can’t be ignored. “At this point in time,” says Persson, “we
need to look at all the possible solutions.” And MXene-like materials are emerging as potentially important candidates.
</item 31>

<item 32>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
Feedbacks calculated from observations or from atmosphere-only model simulations forced with the observed surface warming pattern over the past couple of
decades imply less warming than those from model simulations with a fully interactive ocean, which have the freedom to create their own surface warming patterns
[e.g., Gregory et al., 2020].

The third strand of research came from oceanography, showing that the atmospheric cooling effect of ocean heat uptake differs depending on where it occurs: One
unit of ocean heat uptake in high latitudes cools Earth more effectively than the same unit taken up by the low-latitude oceans. This difference is relevant
because the largest heat uptake by the ocean occurs at higher latitudes. The effect, termed ocean heat uptake efficacy, turns out to be another manifestation of
the dependence of radiative feedbacks on surface temperature patterns [Winton et al., 2010; Lin et al., 2021].
</item 32>

<item 33>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
But he found himself daydreaming of a simpler life, where he could keep bees, grow vegetables and press cider to drink on a Friday night and live closer to the
land.

Visiting the Possibility Alliance, a sprawling 11-acre homestead filled with fruit trees, goats, chickens and a garden, allowed him to scratch an itch he’d
entertained for much of his adult life.

The Hughes family, which operates the homestead, and its guests live largely without electricity or modern technology.

The family of four doesn’t fly because of climate concerns and also doesn't own a car. Their main use of fossil fuels is to take passenger trains to climate
protests.

They try to eschew capitalism and have built, instead, a "gift economy" where resources are shared and skills are traded among neighbors in this tiny corner of
Maine. They grow much of their own food, hold trainings for climate protesters and plan to host refugees as climate disasters worsen.

Nights are illuminated by candlelight. Neighbors drop by without notice.

“We’ve created what existed 100 years ago,” Ethan Hughes said.
</item 33>

<item 34>
Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' (Fortune,
2023-10-10T20:17:09+00:00)
6 insurrection—an event unseen for centuries in American politics. Then there’s the first major European ground war since World War II in Ukraine, not to
mention market crashes in 2008 and 2020 that recall the Great Depression itself. As the first digitally native generation, Gen Z is experiencing it all through
videos, images, and articles online, which is shaping their mental health, workplace attitudes, and financial habits in visible ways.

It’s no wonder, this Gen Z reporter notes, that 46% of young workers ages 18 to 26 say that they are regularly so distraught over what is happening in the news
that they are unable to function at work, according to a 2023 Edelman report. By comparison, 38% of millennials, 24% of Gen Xers, and 19% of baby boomers and
older generations say the same.

Everything about their behavior communicates that Gen Z is just not okay with it.
</item 34>

<item 35>
'We know far more about the deep ocean than the moon or Mars,' says explorer Jon Copley (Live Science: The Most Interesting Articles, 2023-09-28T11:01:16+00:00)
How worried are you about that?

JC: I think it's great that deep sea mining has made people care more about the deep ocean, but it hasn't actually started yet and research does not support
some of the more hyperbolic headlines.

Related: More than 5,000 new species found in 'pristine' deep-sea wilderness. But they could soon be wiped out.

There's a lot of research focused on how we're going to manage mining, if it does go ahead. And there are some habitat types in the deep ocean that we don't
need to do further research on, because we know they are so vulnerable. We know that we would risk species extinction at active hydrothermal vents, for example,
because they're a tiny habitat globally — just 50 square kilometers [19 square miles] — with more than 400 animal species not found in any other habitat type.
</item 35>

<item 36>
'They seemed primed to take over': How the Great Dying doomed the 'beast tooth' and set the stage for the dawn of the dinosaurs (Live Science: The Most
Interesting Articles, 2023-10-01T08:00:25+00:00)
Is this ancient event a possible analog for a sixth, human-caused, climate-change-driven mass extinction today? In answering this question, we will at times
work our way through some details of the science, but the payoff is that we will see not just that scientists are able to unravel such mysteries, but how they
do it.

In the late Proterozoic eon, around 550 million years ago, Earth had thawed out from a series of major glaciations, perhaps even global snowball conditions. The
end of the Proterozoic marked the beginning of a brand new era — the Paleozoic, which extended from around 540 million to 251 million years ago.

The first period of the Paleozoic — the Cambrian — saw a remarkable explosion in the diversity of life, known, appropriately, as the Cambrian explosion.
</item 36>

<item 37>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
In addition to our growing knowledge of the pattern effect, we also have learned that radiative feedbacks depend on global mean temperature itself: Warming
Earth by 1 K from the LGM emphasizes different feedbacks (e.g., the sea ice albedo feedback) than warming by 1 K from a Miocene hothouse world or warming from 4
to 5 K in a high-emission scenario in a century or two from today (e.g., the water vapor feedback [Bloch-Johnson et al., 2021]). The pattern effect and the
feedback temperature dependence add uncertainties to estimates of climate sensitivity based on the paleorecord, but quantifying their effects would make these
records more relevant to constraining climate sensitivity and expected future warming.

Outstanding questions concern how representative pattern changes in the past century, the past millennium, or quasi-equilibrated times millions of years ago are
of expected future changes.
</item 37>

<item 38>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
The reason for the poor predictive power of the climate record from past decades is that the observed surface warming pattern caused feedbacks that were more
stabilizing than the ones projected for the future. The warming was particularly pronounced in the sweet spots of the western Pacific and the subtropical
eastern Pacific—a perfect combination for enhanced radiative damping (Figure 2, left). In particular, low-lying cumulus clouds in the eastern Pacific covered a
larger area and reflected more sunlight back to space than studies suggest they will in the future.

Difficulty in quantifying the pattern effect stems from uncertainties in observed surface temperature trends, the magnitude of ocean heat uptake, and the short
observational record of Earth’s energy budget.

Importantly, the measurement and magnitude of the observed pattern effect are still debated and uncertain.
</item 38>

<item 39>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
Author Information

Maria Rugenstein (maria.rugenstein@colostate.edu), Colorado State University, Fort Collins; Mark Zelinka, Lawrence Livermore National Laboratory, Livermore,
Calif.; Kristopher B. Karnauskas, University of Colorado Boulder; Paulo Ceppi, Imperial College London, London; and Timothy Andrews, Met Office Hadley Centre,
Exeter, U.K.

Citation: Rugenstein, M., M. Zelinka, K. B. Karnauskas, P. Ceppi, and T. Andrews (2023), Patterns of surface warming matter for climate sensitivity, Eos, 104,
https://doi.org/10.1029/2023EO230411. Published on 31 October 2023.

Text © 2023. The authors. CC BY-NC-ND 3.0

Except where otherwise noted, images are subject to copyright. Any reuse without express permission from the copyright owner is prohibited.
</item 39>

<item 40>
‘Zoom fatigue’ is common struggle for remote workers. Here’s how to handle it, according to experts (FOX News - Health, 2023-12-01T11:25:35+00:00)
CAN YOU AVOID END-OF-THE-YEAR CAREER BURNOUT? JOB EXPERTS REVEAL SECRETS

Corcoran agreed, noting that many report feeling livelier, happier and more active during in-person meetings.

"Because video platforms artificially limit our perspective, we miss the cues we’re used to from live, in-person interactions," she told Fox News Digital.

"As social beings, we’re highly attuned to these cues, and it takes more work for us to look for and process them in the virtual environment, leading to the
fatigue we experience."

Even beyond the workplace, Elliott noted that people are using video in more personal contexts than ever before, such as sharing reels on social media and
calling loved ones on FaceTime.

"Because video platforms artificially limit our perspective, we miss the cues we’re used to from live, in-person interactions."

"This can further exacerbate the problem if you're already spending a significant portion of your day on Zoom," he said.
</item 40>

<item 41>
Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' (Fortune,
2023-10-10T20:17:09+00:00)
And last summer, after the Supreme Court overturned Roe v. Wade, a then-19-year-old activist raised over $2 million in abortion funds by trolling Florida Rep.
Matt Gaetz.

It remains to be seen what they will do when Gen Z comes into decision-making positions in the workplace—perhaps they’ll channel their rage and cynicism
effectively as some have already demonstrated—but Abinusawa warned that “a society where the young have little to no hope for the future is not a sustainable
one.”
</item 41>

<item 42>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
Winton, M., K. Takahashi, and I. M. Held (2010), Importance of ocean heat uptake efficacy to transient climate change, J. Clim., 23, 2,333–2,344,
https://doi.org/10.1175/2009JCLI3139.1.

Wills, R. C. J., et al. (2022), Systematic climate model biases in the large-scale patterns of recent sea-surface temperature and sea-level pressure change,
Geophys. Res. Lett., 49, e2022GL100011, https://doi.org/10.1029/2022GL100011.

Zhou, C., et al. (2021), Greater committed warming after accounting for the pattern effect, Nat. Clim. Change, 11, 132–136,
https://doi.org/10.1038/s41558-020-00955-x.
</item 42>

<item 43>
The shipping industry’s boom-and-bust cycle is so severe carriers face going from bumper profits to losing money in the space of a few years: 'I'm certainly
concerned' (Fortune, 2023-11-12T17:51:37+00:00)
Transiting the Suez Canal from Asia next year, for instance, will be 15% costlier, the waterway’s authority said in mid-October without explanation. On the
other main trade route, ships passing through the drought-stricken Panama Canal are facing long waits, surcharges, and time- and fuel-consuming detours around
South America to avoid the delays.

Those costs are small compared with the $1 trillion in investment the industry faces in the coming decades to decarbonize — a shift that will require engines
that run on cleaner-burning fuels and new infrastructure to produce, store and transport the alternative fuels.

The big European carriers have issued estimates for the surcharges that’ll take effect with shipping’s upcoming entry into the European Union’s Emissions
Trading System in January.

At CMA CGM, one of the strategies to cushion the peaks and valleys of shipping is to diversify.
</item 43>

<item 44>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
Scholars have argued that putting too much emphasis on the platforms can be too simplistic and reeks of technological determinism; they have also pointed out
that cultures like the United States’ and the legacy media have a long history with post-truth. That said, there are certainly dynamics going on that we did not
anticipate, and we don’t seem quite sure what to do with them, even with multiple areas of scholarship in communication, disinformation, and social media and
democracy working on these inquiries for years.

What seems clear is that the Internet is not the connectedness we imagined. The ecology and spirituality of the 1960s, which shaped and structured much of what
we see as energy democracy and the good future today, told us we were all connected. Globally networked — it sounds familiar, like a fevered dream from the
1980s or 1990s, a dream that in turn had its roots in the 1960s and before.
</item 44>

<item 45>
New ultrathin materials can pull climate-warming CO2 from the air (Science News For Students, 2023-12-15T11:30:00+00:00)
This is another in our series of stories identifying new technologies and actions that can slow climate change, reduce its impacts or help communities cope with
a rapidly changing world.

One up-and-coming class of tiny materials could help solve the huge problem of climate change. How? By catching climate-warming carbon dioxide out of the air.

The nano-scale materials are called MXenes (MAX-eenz). Their name comes from their structure, which is made up of alternating layers of M and X atoms. “M”
layers consist of some transition metal. (Those atoms are bulky and can shed a variable number of electrons.) “X” layers contain atoms of another element, such
as carbon or nitrogen.

A single MXene sheet contains a few of these alternating M and X layers chemically bound to each other. It may look like MXM or MXMXM. Each sheet is typically
only about 1 nanometer thick. For comparison, a sheet of paper is 100,000 times thicker.
</item 45>

<item 46>
La desigualdad del estrés por calor (Eos: Earth And Space Science News, 2023-09-26T12:15:35+00:00)
“Zonas con menos árboles en las calles, menos acceso a espacios verdes—Allí es donde se comienza a ver el legado de la desinversión y el racismo institucional
que ha entrado en juego”.

“Zonas con menos árboles en las calles, menos acceso a espacios verdes—Allí es donde se comienza a ver el legado de la desinversión y el racismo institucional
que ha entrado en juego”, dijo Lara Whitely Binder, gerente del programa de preparación climática del condado de King County, Washington, quien no estuvo
involucrada en la investigación.

Debido a que el clima está cambiando y avanzando, de ahora en adelante las ciudades deben prepararse para veranos más calientes y olas de calor mortales más
frecuentes usando distintos medios, tanto en los interiores como al aire libre.
</item 46>

<item 47>
La desigualdad del estrés por calor (Eos: Earth And Space Science News, 2023-09-26T12:15:35+00:00)
En vecindarios sin parques ni árboles grandes, donde el asfalto y los edificios absorben e irradian el calor del verano, a los residentes les puede resultar
difícil escapar de las temperaturas extremas tanto en interiores como en el exterior. Este tipo de exposición continua puede ser muy peligrosa. El cuerpo se
vuelve incapaz de enfriarse adecuadamente mediante la sudoración. La exposición crónica al calor excesivo tiene un impacto en la salud cardiovascular,
respiratoria y mental, dijo Tummala. “La exposición prolongada a temperaturas nocturnas elevadas puede afectar la calidad del sueño y la salud mental y
cardiaca,” dijo. “Estás constantemente a una temperatura elevada con la que tu cuerpo tiene que lidiar”.
</item 47>

<item 48>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
The first strand came from analyses of climate feedbacks and sensitivity in model simulations of unequilibrated, transient climate change. If feedbacks were
constant, the expected equilibrium temperature change (climate sensitivity) could be estimated using a very simple energy balance model that linearly
extrapolates the relationship between global temperature change and radiative imbalance [e.g., Gregory et al., 2004]. When longer, fully equilibrated
simulations became available, it became evident that the simple estimation methods assuming constant feedbacks systematically underestimate the actual
equilibrium climate sensitivity. The reason for this underestimation is indeed the evolution of the surface warming pattern, which initially emphasizes more
stabilizing radiative feedbacks but later, during equilibration, emphasizes less stabilizing radiative feedbacks [e.g., Senior and Mitchell, 2000; Rugenstein et
al., 2020].

The second strand related the idea of constant feedbacks to the efforts of estimating equilibrium climate sensitivity from the historical record, as mentioned
above.
</item 48>

<item 49>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
Kalmus, trailed by a reporter, buzzed around the homestead like a giddy kid, hiking back and forth to a home in an eco-village near the Possibility Alliance
where he was living and working on his research. He tramped down a forest trail in bare feet, paused work for a midday dip in a shady stream and never changed
his T-shirt.

He ground his coffee with a mortar and pestle, thumbed through a book of Enneagram personality types on the porch with his son Zane and listened to Ethan’s
stories about sailing across the Atlantic in a cargo ship to avoid the carbon emissions of flying.

“I feel met by Peter,” Sarah Wilcox-Hughes said of Kalmus. “I don’t feel like there are many people prioritizing the needs of the planet as he is and we are.”

Kalmus found life in Belfast, and at the Possibility Alliance, slower, richer and more connected to the natural world.
</item 49>

<item 50>
‘Zoom fatigue’ is common struggle for remote workers. Here’s how to handle it, according to experts (FOX News - Health, 2023-12-01T11:25:35+00:00)
CLICK HERE TO SIGN UP FOR OUR HEALTH NEWSLETTER

"Whenever you’re able, suggest audio-only meetings that don’t require screens so people can do things they would naturally do in an in-person setting, like move
around or look out a window," she advised.

"Walking away from our screens can reduce the urge to multitask, reduce eye fatigue and increase our capacity to move around."

Turning off the self-view option can also be helpful, Corcoran said — "it can be particularly distracting and draining because we’re not used to seeing our own
images so much."

As the use of video platforms is still relatively new for many professionals, Corcoran noted that the long-term effects aren’t yet known.

CLICK HERE TO GET THE FOX NEWS APP

"That being said, we know that constant feelings of fatigue, exhaustion and physical strain or tension do correlate with negative mental health outcomes," she
said.

"It’s important to take steps to reduce Zoom fatigue before it leads to things like chronic stress or burnout."
</item 50>

<item 51>
'We know far more about the deep ocean than the moon or Mars,' says explorer Jon Copley (Live Science: The Most Interesting Articles, 2023-09-28T11:01:16+00:00)
But I'm confident that we will see protection for active hydrothermal vents, because we scientists have been saying that for years.

Hydrothermal vents are some of the most vulnerable deep sea habitats and home to 400 animal species not found anywhere else. (Image credit: Universal History
Archive/Universal Images Group via Getty Images)

SP: Deep sea mining is perhaps more manageable in terms of its impacts than other human activities. If not mining, what is the biggest human threat to the deep
sea?

JC: To my mind, it's climate change . And it affects the deep ocean in lots of different ways. The one that concerns me particularly is deoxygenation — the
reduction in oxygen levels — because deep sea animals need oxygen and they get it from the seawater.

Oxygen is carried down by currents that form in the polar regions and sink and spread throughout the deep ocean. As a result of climate change, the ocean is
getting warmer and that means it can't carry as much dissolved oxygen .
</item 51>

<item 52>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
However, they also depend on the spatial pattern of surface warming: Much like applying a force uniformly over someone’s entire body will elicit a very
different reaction than tickling the soles of that person’s feet, a degree of global warming spread out evenly will cause a different radiative response than if
that same warming were concentrated in a climate sweet spot (a location where surface warming produces efficient radiative damping).

A wide variety of processes affect the evolution of surface temperature, from greenhouse gas forcing and regional aerosol forcing to natural oscillations
involving the ocean and atmosphere to the continental boundary conditions and the extent of ice sheets and sea ice. The pattern of surface temperature change
over the past 40 or so years featured a pronounced spatial structure, with some locations even cooling in spite of the global mean warming on the order of 1 K
(Figure 2, bottom left).

Fig. 2. At left is the surface and vertical structure of warming observed over the past few decades in the tropical Pacific.
</item 52>

<item 53>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
Rugenstein, M., et al. (2020), Equilibrium climate sensitivity estimated by equilibrating climate models, Geophys. Res. Lett., 47, e2019GL083898,
https://doi.org/10.1029/2019GL083898.

Senior, C. A., and J. F. B. Mitchell (2000), The time-dependence of climate sensitivity, Geophys. Res. Lett., 27, 2,685–2,688,
https://doi.org/10.1029/2000GL011373.

Sherwood, S. C., et al. (2020), An assessment of Earth’s climate sensitivity using multiple lines of evidence, Rev. Geophys., 58, e2019RG000678,
https://doi.org/10.1029/2019RG000678.
</item 53>

<item 54>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
While on his working vacation in Maine, Kalmus invited NBC News for a short visit to the community to see the crisis through the eyes of Americans taking it as
seriously as anyone.

The Possibility Alliance is an 11-acre homestead that operates without power, running water or a sewer system. The Hughes family, which operates the alliance,
pays only one monthly bill for landline phone service. (Evan Bush / NBC News)

Detoxifying

About a decade ago, Kalmus ditched a promising academic career in astrophysics to study the physics of climate change instead.

He made other changes too. He stopped flying, became a vegetarian and ditched gasoline-powered cars (he drives a Tesla), cutting his personal emissions by about
90%, according to his math.

It wasn’t a political statement, but a detoxification.

“Burning fossil fuels started to feel more and more disgusting,” he said.

Ethan Hughes, who operates the Possibility Alliance, rejects globalization and thinks capitalism is a ruinous symptom of “greed and separation.”
</item 54>

<item 55>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
During his time in Belfast, Kalmus strummed a mandolin in a Celtic folk music group that meets at the eco-village, along with his son Zane, Wilcox-Hughes and
her daughter, Etta. He picked berries and ate fresh goat cheese and helped Ethan load hay into a neighbor’s barn.

climate scientist music group (Evan Bush / NBC News)

He spoke at the local library about climate change. He shared art, food and labor with neighbors. He saw how a community was sprouting up around the Possibility
Alliance — people willing to sacrifice the conveniences of a modern life built on fossil fuels.

He felt safer here and more energized to work.

It wasn’t about the place, but the people, he realized.

People at the center

Maine, like anywhere else, isn’t immune to climate change. But here, at least, he had people who understood his angst, who could share music and joy, who were
willing to make their own sacrifices, and who were trying to prepare their community in the way they thought was best.
</item 55>

<item 56>
10 scorching-hot discoveries made about the sun in 2023 (Live Science: The Most Interesting Articles, 2023-12-24T14:00:00+00:00)
"A similar solar storm today would be catastrophic for our modern technological society," the researchers wrote.

Ancient 'superflares' may have sparked life on Earth

Ancient "superflares" could have sparked life on Earth, researchers claim. (Image credit: NASA's Goddard Space Flight Center)

Continuing the ancient solar storm trend, a May study revealed that "superflares" from the early hyperactive sun could have provided the energy needed to spark
life on Earth billions of years ago.

Researchers recreated Earth's early atmosphere in the lab and fired charged particles, like those found in solar wind, at the primitive gasses and found that
this created both amino acids and carboxylic acids — building blocks for proteins and all organic life.

Similar experiments recreating lightning have also created these compounds in the lab. But the team argues that lightning strikes would not have provided
anywhere near as much power as superflares, which makes them a less suitable candidate for kickstarting life on Earth.
</item 56>

<item 57>
La desigualdad del estrés por calor (Eos: Earth And Space Science News, 2023-09-26T12:15:35+00:00)
El peligro del calor implacable

Las islas de calor urbano, zonas calientes de la ciudad donde el asfalto, alta densidad de edificios e infraestructura causan que la temperatura aumente más que
en las áreas circundantes, albergan millones de personas que no son capaces de escapar de los efectos del implacable calor del verano.

La exposición crónica al calor excesivo tiene un impacto en la salud cardiovascular, respiratoria y mental.

“Las personas que viven en zonas urbanas suelen caminar para hacer sus tareas diarias. Una de las ventajas de vivir en una ciudad es que las cosas están cerca y
puedes utilizar el transporte público, pero eso también significa que tienes que pasar más tiempo al aire libre”, dijo Neelima Tummala, médico y co-directora
del Instituto de Salud Ambiental en la Universidad George Washington, quien no estuvo involucrada en el estudio.
</item 57>

<item 58>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
The government calls for embedding resilience, sustainability, and nature-based solutions in infrastructure planning and investment to enhance climate
resilience. Likewise, investments in renewable energy are expected not only to enhance energy security but also reduce greenhouse gas (GHG) emissions.

From an economic vantage point, GHG emissions are economic “bads” and are a cause of inefficiency as they get over-produced. Their effects, however, go beyond
national borders, and while the Philippines contributes minimally to global GHG emissions, the World Risk Index 2022 report identifies the Philippines as the
most disaster-prone country in the world.

The government envisions that by 2028: (1) Climate and disaster risk resilience of communities and institutions will increase (2) Ecosystem resilience will be
enhanced, and (3) A low-carbon economy transition is enabled.

To ensure that these are realized, the government is set to safeguard cross-sectoral convergence and implement a comprehensive risk management approach to
address the adverse consequences posed by climate change.
</item 58>

<item 59>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
The most pressing question is whether climate model simulations will be as far off from observations in the future as they have been relative to recent past
conditions (compare Figure 2, left and right). If this turns out to be the case, how will model biases in surface warming trends be reflected in radiative
feedbacks and global mean warming rates? We need to quantify whether coupled models compensate for their bias in the surface warming pattern and radiative
feedbacks through erroneous ocean heat uptake rates or aerosol forcings.

The advent of global coupled storm- and cloud-resolving models, which currently can simulate only a few months to years at a time, presents an exciting
opportunity. How do we evaluate surface warming patterns and radiative feedbacks in these simulations and compare them meaningfully to observations and coarser
resolution models, knowing that even several decades are not enough to robustly detect a forced trend in some regions?
</item 59>

<item 60>
'We know far more about the deep ocean than the moon or Mars,' says explorer Jon Copley (Live Science: The Most Interesting Articles, 2023-09-28T11:01:16+00:00)
SP: Some of the chapters focus on dispelling misconceptions people might have about what's down there in the ocean. What, to your mind, is the biggest, most
pervasive myth about the deep sea?

JC: It's the idea that we know almost nothing about it. There's this very popular idea that we know more about the moon or Mars than the deep ocean. That's only
really true for one very specific aspect of knowledge — having a detailed map of the terrain of its solid surface — because the moon and Mars are not covered in
seawater, which blocks radar and means we have to use sonar in the deep ocean. Apart from that, we know far more about the deep ocean than those other places.

SP: The deep sea has attracted a lot of attention recently in the advent of deep sea mining . How worried are you about that?
</item 60>

<item 61>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
Gregory, J. M., et al. (2004), A new method for diagnosing radiative forcing and climate sensitivity, Geophys. Res. Lett., 31, L03205,
https://doi.org/10.1029/2003GL018747.

Gregory, J. M., et al. (2020), How accurately can the climate sensitivity to CO 2 be estimated from historical climate change?, Clim. Dyn., 54, 129–157,
https://doi.org/10.1007/s00382-019-04991-y.

Lin, Y., et al. (2021), The dominant contribution of Southern Ocean heat uptake to time-evolving radiative feedback in CESM, Geophys. Res. Lett., 48,
e2021GL093302, https://doi.org/10.1029/2021GL093302.

Rugenstein, M., et al.
</item 61>

<item 62>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
The paleorecord could further be crucial for understanding the timescales and relative importance of internal variability, forced response, and extratropical
forcing to rates of warming or cooling in the equatorial Pacific in nature and in climate models.

Should We Trust Model Patterns in Climate Change Scenarios?

Our understanding of the pattern effect raises the question of whether climate models can reproduce observed warming patterns. Coupled climate models simulate a
freely evolving ocean surface and hence have their own expression of internal variability. For example, we would not expect a coupled model to simulate El Niño
events at the same time as they happened in nature, except by chance. This feature complicates comparisons between models and observations, especially on
decadal timescales and beyond, for which we know little about the spectrum of internal variability in the real world.

Unfortunately, coupled climate models seem unable to simulate observed surface warming patterns across some key regions in which surface sensitivity modulates
clouds, even when they account for internal variability.
</item 62>

<item 63>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
The other end of the uncertainty range implies that the pattern effect might have been negligible over the past century and that it might apply only over
shorter timescales or in drastically different climate states.

Difficulty in quantifying the pattern effect stems from uncertainties in observed surface temperature trends, the magnitude of ocean heat uptake, and the short
observational record of Earth’s energy budget. Our incomplete quantitative understanding of how clouds react to their environment and how these dependences are
represented in the highly parameterized climate models further limits our ability to quantify the full range of the possible and recently realized pattern
effect.

Pressing, fundamental questions for climate dynamics concern how surface temperature patterns come about, how Earth’s radiation budget depends on the details of
the surface temperature patterns, and the extent to which the two depend on each other.
</item 63>

<item 64>
The housing winter arrives early as existing home sales crash to Great Recession levels and the long-feared ‘deep freeze’ sets in (Fortune,
2023-10-20T20:30:29+00:00)
But this isn’t like 2008 or 2010 again. It’s time for a 1980s history lesson.

Back to the 1980s future

The housing market today isn’t identical to that of the ‘80s, but it’s pretty close. In a lot of ways, millennials are being forced to follow the housing
journey of their boomer parents as they face a frozen, unaffordable market with rising interest rates, as noted recently by both BofA Research economist Jeseo
Park and by First American chief economist Mark Fleming.

Essentially, millennials are a giant generation all collectively coming of homebuying age nearly simultaneously—just like their boomer parents in the 1980s.
They’re the “biggest share of the “homebuying pie,” as Redfin puts it, purchasing about 60% of homes bought with mortgages during the past few years.
</item 64>

<item 65>
'They seemed primed to take over': How the Great Dying doomed the 'beast tooth' and set the stage for the dawn of the dinosaurs (Live Science: The Most
Interesting Articles, 2023-10-01T08:00:25+00:00)
Inostrancevia is a genus from the extinct group Theriodontia that appeared during the Middle Permian. (Image credit: Stocktrek Images/Getty Images)

The excerpt below is taken from "Our Fragile Moment: How Lessons from Earth's Past Can Help Us Survive the Climate Crisis" (Hachette Book Group, 2023), by
Michael Mann. It looks at how climate change following the Cambrian explosion caused the biggest mass extinction on Earth — dooming the creatures set to
dominate and set the stage for dinosaurs to rule.

The mechanisms that can freeze the planet, as was the case with Snowball Earth can also lead to inhospitably hot climates, when enough carbon dioxide enters the
atmosphere. Arguably the greatest extinction event of all time — called the Great Dying — appears to have resulted, at least in part, from a massive
heat-inducing release of carbon into the atmosphere 250 million years ago.
</item 65>

<item 66>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
To understand the dynamic, we need to take a closer look at the concept of energy democracy, as generally understood by the climate movement, and its tenets:
renewable, small-scale systems, and community control. The bitter irony of the current moment is that it’s not just rising authoritarianism that is blocking us
from good futures. It’s also our narrow and warped conceptions of democracy that are trapping us.

This article contains affilate links; if you click such a link and make a purchase, we may earn a commission.
</item 66>

<item 67>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
He has been on a 15-year quest to halt climate change and has already oriented most everything in his life to minimize his own climate footprint and maximize
what he can do to push society to change.

Nothing has been enough.

And so this summer, as extreme heat and rainfall battered many American communities, Kalmus visited Maine searching for something different. He was looking not
for climate solutions, but for new ways to withstand climate change’s harms — and also to make peace with the grief that often feels inescapable for those who
reckon with global warming daily.

climate scientist exercise bicycle environmentalist (Evan Bush / NBC News)

That search brought him to an unusual community called the Possibility Alliance, a homestead where a small group of people is trying to figure out how to live
in a world headed for a breakdown (as they see it).

Its leaders eschew capitalism, grow much of their own food, and live without electricity. They’ve dedicated their lives to protest, living off the land as
simply as possible, and are preparing their community for ecological disaster and to host climate refugees.
</item 67>

<item 68>
A new hydrogel could help pull drinking water from the air (Science News For Students, 2023-11-03T10:30:00+00:00)
There was a problem signing you up.

More food, less water

Earth’s freshwater supply is falling due to climate change. By 2050, more than five billion people are expected to lack water at least one month per year. Some
shortages are due to warmer temperatures and less snow to feed into rivers. That puts stress on supplies of water for drinking and growing crops.

Whatever the cause, better water-slurping gels could make it easier to grow food in such water-stressed places — including those plagued by drought.

Tan’s team has developed a gel-based device called the SmartFarm. This tiny greenhouse with a moving roof grows crops without having to add water. In the
evening, the top opens. Gels inside capture water from the night air. A solar-powered motor closes the roof during the day. When warmed by the sun, the gel
releases its water. Tan’s team used the SmartFarm to grow a leafy green vegetable called Ipomoea aquatica.
</item 68>

<item 69>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
In other words, first, we need to improve our understanding of drivers of sea surface temperature patterns—decadal coupled variability; the pace and spatial
structure of ocean heat uptake; and forcing by aerosols, greenhouse gases, and volcanoes—as well as the relative timing of these drivers. An important goal is
to explain historical patterns quantitatively and to predict their likely future evolution. Second, we need to quantify the dependence of local and remote
top-of-the-atmosphere radiative fluxes on the magnitude, spatial scale, and sign (positive or negative) of surface temperature changes. Third, we have to
explain how ocean heat uptake and radiative feedback are connected on various timescales, both globally and regionally.

Interpreting Paleorecords Requires Caution

In the same way that the historical evolution of surface warming and the planet’s radiation budget inform Earth’s future only to a limited extent, analyses of
paleorecords for recent and deep time intervals must account for the pattern effect to be applicable to climate projections of the future.
</item 69>

<item 70>
Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' (Fortune,
2023-10-10T20:17:09+00:00)
Everything about their behavior communicates that Gen Z is just not okay with it. This ranges from their widespread, hell-bent determination to find purpose in
work and pushing their employers to have a social conscience, to a sense of despair over their own and the world’s future finances. They have largely given up
on saving money and instead dish out for little “treats” as a way to cope with the larger absurdity of 21st-century life.

Consider the lifetime that was three years ago, as Gen Z emerged into young adulthood, when online videos of the murder of George Floyd shook the country in May
2020, resulting in a summer of violent Black Lives Matter protests and riots. A year later, people watched as armed right-wing extremists stormed the Capitol on
Jan. 6, 2021, following the election defeat of former President Donald Trump.
</item 70>

<item 71>
Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' (Fortune,
2023-10-10T20:17:09+00:00)
workforce, but they’re already radically redefining the meaning of work.

Gen Zers want a sense of purpose, so they prioritize environmental, social, and governance (ESG) in the workplace, which encompasses their companies taking
accountability for their sustainability and environmental impacts, providing education and awareness for social issues, and ensuring diverse and inclusive
boards and teams.

And it tracks: Roughly two-thirds of Gen Zers say they frequently speak about important societal issues while at work, according to the Edelman report. They’re
also influencing their older coworkers when it comes to areas like work-life balance, fair pay, and employers’ involvement on social issues.

Gen Z’s financial stress

But don’t forget, Gen Zers have also lived through a global pandemic that shuttered the world for nearly two years, two recessions, and a mounting student debt
crisis, leaving them with little savings but an abundance of financial despair.
</item 71>

<item 72>
La desigualdad del estrés por calor (Eos: Earth And Space Science News, 2023-09-26T12:15:35+00:00)
This is an authorized translation of an Eos article. Esta es una traducción al español autorizada de un artículo de Eos.

Cuando cada verano temperaturas récord y domos de calor envuelven amplias zonas de los Estados Unidos, las personas experimentan estos eventos de calor extremo
de manera diferente. Aquellos que viven en vecindarios históricamente marginalizados, donde el uso de la tierra y las políticas de vivienda discriminatorias
hicieron que la segregación y el racismo florecieran, al día de hoy aún presentan mayor riesgo por las temperaturas altas y los efectos del estrés por calor en
la salud.

En un nuevo estudio publicado por One Earth, investigadores demostraron que en 481 ciudades estadounidenses el estrés por calor afecta a las personas pobres y
de color desproporcionalmente.
</item 72>

<item 73>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
One of the grand challenges in climate science is to reduce uncertainty in estimates of climate sensitivity, which quantifies how much Earth’s surface warms in
response to a doubling of carbon dioxide relative to preindustrial levels. This uncertainty is large because climate sensitivity aggregates myriad processes,
from microscale aerosol-cloud interactions to planetary-scale atmospheric and ocean circulations, into one number. Clouds, which are notoriously difficult to
measure and simulate, are the main driver of the uncertainty.

Various lines of evidence are used to estimate climate sensitivity, including climate model simulations of varying complexity, observations over the past
century, proxies that measure climate change in the distant past, and theory. The likely range of estimates of climate sensitivity was stubbornly constant at a
distressingly imprecise 1.5–4.5 K for several decades, but the research community’s efforts have recently chipped away at this range (Figure 1).
</item 73>

<item 74>
Consistent policy is key to unlocking the climate tech boom, but the UK’s U-turn isn’t helping (TechCrunch, 2023-10-13T17:00:20+00:00)
If giants like Ford are feeling the frustration, imagine the mood at a small green tech startup.

So what’s the play? How can the government support those striving against an increasingly insecure (and insincere) backdrop?

The response of any tech company impacted by last week’s news surely has to be this.

First, we need a consistent macroeconomic policy. This has a major impact on startups raising and deploying capital into climate R&D. The global economic
contraction has made it difficult for these businesses to raise any money. Overall approaches to the economy have trickle-down effects on how we, as businesses,
make money and run our operations. Right down to salaries, which alone are hard to keep up with inflation-based pay rises.

The second is a consistent tax policy. One of the most critical things for startups is research and development (R&D) tax credits.
</item 74>

<item 75>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
Res. Atmos., 127, e2022JD036675, https://doi.org/10.1029/2022JD036675.

Bloch-Johnson, J., et al. (2021), Climate sensitivity increases under higher CO 2 levels due to feedback temperature dependence, Geophys. Res. Lett., 48,
e2020GL089074, https://doi.org/10.1029/2020GL089074.

Forster, P., et al. (2021), The Earth’s energy budget, climate feedbacks, and climate sensitivity, in Climate Change 2021: The Physical Science Basis. Working
Group I Contribution to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change, edited by V. Masson-Delmotte et al., pp. 923–1,054,
Cambridge Univ. Press, New York, https://doi.org/10.1017/9781009157896.009.
</item 75>

<item 76>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
Published by MIT Press. Copyright © 2023. All rights reserved.

Burning hills and glowing red skies, stone-dry riverbeds, expanses of brown water engulfing tiny human rooftops. This is the setting for the twenty-first
century. What is the plot? For many of us working on climate and energy, the story of this century is about making the energy transition happen. This is when we
completely transform both energy and land use in order to avoid the most devastating impacts of climate change — or fail to.

Confronting authoritarianism is even more urgent. About four billion people, or 54 percent of the world, in ninety-five countries, live under tyranny in fully
authoritarian or competitive authoritarian regimes. The twenty-first century is also about the struggle against new and rising forms of authoritarianism.
</item 76>

<item 77>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
Authoritarians know this, and this is why they have troll farms that can manipulate the range of solutions and the sentiments about them.

The Internet as we experience it represents a central obstacle to climate action, through several mechanisms. Promotion of false information about climate
change is only one of them. There’s general political polarization, which inhibits the coalitions we need to build to realize clean energy, as well as creates
paralyzing infighting within the climate movement about strategies, which the platforms benefit from. There’s networked opposition to the infrastructure we need
for the energy transition. There’s the constant distraction from the climate crisis, in the form of the churning scandals of the day, in an attention economy
where all topics compete for mental energy. And there’s the drain of time and attention spent on these platforms rather than in real-world actions.

Any of these areas are worth spending time on, but this essay focuses on how the contemporary media ecology interferes with climate strategy and infrastructure
in particular.
</item 77>

<item 78>
Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' (Fortune,
2023-10-10T20:17:09+00:00)
That’s why Gen Z is more concerned with sustainability than any generation before them. Just look at Greta Thunberg: The 20-year-old has become one of the
best-known environmental activists, famously speaking at the United Nations in 2019 with scathing words for world leaders:

“You have stolen my dreams and my childhood with your empty words. And yet I’m one of the lucky ones,” Thunberg said. “People are suffering. People are dying.
Entire ecosystems are collapsing. We are in the beginning of a mass extinction, and all you can talk about is money and fairy tales of eternal economic growth.
How dare you!”

Gen Z in the workplace

By 2030, Gen Z will account for nearly one-third of the U.S. workforce, but they’re already radically redefining the meaning of work.
</item 78>

<item 79>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
“Everything is already spinning out of control,” said Sarah Wilcox-Hughes, dismayed.

Kalmus opened up to the table. He’d been procrastinating for months on finishing the research and he was running into his “psychological limitations.” He felt
he was letting the planet down.

“Anyone want to rub Peter’s shoulders as he writes?” Ethan said.

Later, Kalmus relayed a recent, gloomy conversation with a fellow scientist.

“The Amazon is probably past its tipping point,” Kalmus said. “And I can’t believe that it happened during my lifetime.”

Peter Kalmus sits with his son Zane on a porch swing at the Possibility Alliance. (Evan Bush / NBC News)

Crickets trilled. Roosters crowed. A tear dripped down Ethan’s cheek.
</item 79>

<item 80>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
He’s trying to build a gift economy in Belfast, in which neighbors share resources and services with no expectation of repayment. “I’m trying to role-model
post-petrol,” Hughes said. (Evan Bush / NBC News)

Kalmus, now a married father of two teenagers, is introspective, earnest and narrowly focused on climate change in both his personal and professional life. Tall
with an athletic build, a neat beard and a tangle of messy hair that has a skunklike streak of white, he can sometimes come off like an overgrown teenager: He’s
the smartest person in the room, brimming with idealism and still indignant — with a zeal many adults can no longer muster — about how life’s most annoying
systems work (like the U.S. medical system, a symptom of what he calls the “capitalist death cult”).

The changes he made weren’t enough. Kalmus wrote a book on climate change and chained himself to a Chase bank in protest of fossil fuel financing. Twice, he’s
been arrested during climate protests.
</item 80>

<item 81>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
Strong warming in regions of deep convection such as the western Pacific is communicated throughout the troposphere and leads to strong warming aloft, enhancing
radiative emission to space. In contrast, parts of the eastern Pacific have cooled. Warming aloft and cooling at the surface enhance the lower tropospheric
inversion strengths in the eastern Pacific and promote extensive shallow cloud cover that reflects solar radiation and keeps Earth relatively cool. At right is
a typical representation of climate change simulated with coupled climate models, which create their own surface warming pattern that differs from the observed
one. In coupled model simulations, warming aloft is less pronounced, and relative to the observed case, the lower tropospheric inversion strength is decreased,
and cloud cover is reduced through time. Both factors lead to less efficient cooling near Earth’s surface in models than in nature.

Feedbacks involving clouds and the atmospheric temperature structure are most sensitive to spatial differences in warming.
</item 81>

<item 82>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
The searing temperatures killed a dogwood tree in his front yard. His productivity slowed — he couldn’t focus on science.

Kalmus had dreamed of making a life in the Pacific Northwest, which he felt might be buffered from the worst of climate change. That same year, a heat wave —
considered nearly impossible if not for the influence of climate change — seared the region for three days, killing hundreds of people, buckling roadways and
sending overheated baby birds jumping from their nests and to their deaths.

“That’s when I realized that no place was safe,” Kalmus said. His family relocated to North Carolina, for his wife Sharon’s work, but the experience planted a
seed:

If he couldn’t stop climate change and he couldn’t avoid it, could he at least find a better way to survive it?

Life at the homestead

Kalmus knew what would not work.

“I very quickly eliminated the idea of being a prepper — you know, hoarding beans and ammunition,” he said.
</item 82>

<item 83>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
Acknowledgments

The authors thank all participants of the US CLIVAR pattern effect workshop. M.Z.’s work was supported by the U.S. Department of Energy (DOE) Regional and
Global Model Analysis program area and was performed under the auspices of DOE by Lawrence Livermore National Laboratory under contract DE-AC52-07NA27344. P.C.
was supported by the UK Natural Environment Research Council, grants NE/T006250/1 and NE/V012045/1. T.A. was supported by the Met Office Hadley Centre Climate
Programme funded by the Department of Business, Energy and Industrial Strategy and received funding from the European Union’s Horizon 2020 research and
innovation program under grant agreement 820829. M.R. was supported by NASA under grant 80NSSC21K1042.

References

Andrews, T., et al. (2022), On the effect of historical SST patterns on radiative feedback, J. Geophys. Res.
</item 83>

<item 84>
'We know far more about the deep ocean than the moon or Mars,' says explorer Jon Copley (Live Science: The Most Interesting Articles, 2023-09-28T11:01:16+00:00)
When water is warmer, the metabolism of things living in the water runs faster and they use up oxygen more quickly, so that makes the problem even worse. And
thirdly, we know that the currents carrying oxygen down to the deep ocean are weakening , because melting ice sheets are making the water fresher and blocking
the formation of dense water than sinks.

Those currents take centuries to complete their journey, which means the changes we have already made are going to carry on being felt for centuries. The deep
ocean is already on track to have 10% less oxygen overall globally than it did in preindustrial times by 2400. It's hard to predict what the knock-on effects
are going to be, but they are going to be widespread and they are coming.

SP: You dedicate much of your time to communicating deep sea science with lay audiences. Why is that so important to you?

JC: I enjoy talking to people about the deep sea because it's not somewhere we think about every day.
</item 84>

<item 85>
The world came tantalizingly close to a deal to phase out fossil fuels (The Verge, 2023-12-13T07:27:19+00:00)
It calls for the tripling of renewable energy capacity globally by 2030, something that more than 100 countries had already pledged to do while negotiations
were taking place last week. Leading up to the conference, the world’s biggest greenhouse gas polluters — the US and China — committed to working toward that
goal together when each country’s climate envoys met in California in November.

The science is clear

There’s a lot of science underpinning these kinds of negotiations. The world, on average, is about 1.2 degrees Celsius hotter than it was before the industrial
revolution. That might not sound like much, but it’s enough to drive more devastating hurricane and wildfire seasons, and suck drought-vulnerable places dry
while displacing other communities faced with rising tides.
</item 85>

<item 86>
The housing winter arrives early as existing home sales crash to Great Recession levels and the long-feared ‘deep freeze’ sets in (Fortune,
2023-10-20T20:30:29+00:00)
housing market set off the dominoes that became the Great Financial Crisis that defined a whole decade economically—and in some ways, our current predicament.

In 2010, the last time existing home sales were so low, the global economy was on life support and the Fed slashed interest rates down to zero to revive the
flatlining patient—fueling an “everything bubble” that inflated all kinds of assets along the way, not the least, housing. The zero-rate regime persisted until
the highest inflation in 40 years surprised many economists and prompted the massive series of rate hikes that have brought us to the once-unthinkable 8%
mortgage.

Some onlookers saw what was coming, for instance Zillow, which warned in May that housing would enter a “deep freeze” if the debt limit standoff failed to
resolve and America defaulted, as that would send mortgage rates up to 8%. Of course, the default was averted, but here we have arrived at 8% mortgage rates
anyway.
</item 86>

<item 87>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
Feedbacks involving clouds and the atmospheric temperature structure are most sensitive to spatial differences in warming. Deep convection in the warmest
tropical regions readily communicates surface conditions upward throughout the troposphere (up to about 10–15 kilometers) and then horizontally across much of
the globe, making the western Pacific a climate sweet spot. This warmer air sitting atop the relatively cool waters in the eastern Pacific or Southern Ocean
acts to stabilize the lowermost troposphere, allowing more extensive low-lying stratus and stratocumulus clouds to develop. Because of their location and
structure, these low clouds efficiently cool the planet and offset some of the initial warming (Figure 2, top left).

Approaching the Problem from Different Angles

Three strands of research have converged over the past few years, highlighting that understanding the pattern effect benefits from contributions from virtually
all climate research communities.

Historically, three strands of research have highlighted the dependence of radiative feedbacks on the surface warming pattern.
</item 87>

<item 88>
La desigualdad del estrés por calor (Eos: Earth And Space Science News, 2023-09-26T12:15:35+00:00)
“Este resultado fue muy interesante – que este nivel de desigualdad y segregación se correlacionara tan fuertemente con los niveles de disparidad de calor,”
dijo.

Proteger a los más vulnerables

“Estudios como este, que buscan comprender más a fondo las disparidades existentes en la exposición al calor, son importantes para identificar qué comunidades
corren mayor riesgo de sufrir cambios ambientales relacionados al cambio climático, como el calor extremo que cada vez empeora más”, dijo Tummala. Se necesitan
medidas de salud pública y estrategias de mitigación para proteger a los ciudadanos con mayor riesgo de sufrir calor – especialmente cuando el aumento de las
temperaturas se vuelve más común.
</item 88>

<item 89>
La desigualdad del estrés por calor (Eos: Earth And Space Science News, 2023-09-26T12:15:35+00:00)
Any reuse without express permission from the copyright owner is prohibited.
</item 89>

<item 90>
'We know far more about the deep ocean than the moon or Mars,' says explorer Jon Copley (Live Science: The Most Interesting Articles, 2023-09-28T11:01:16+00:00)
The shorter format is an opportunity to focus and update the information — there have been quite a few discoveries in lots of different aspects of deep sea
biology since I wrote "Ask an Ocean Explorer."

We know far more about the deep ocean and its inhabitants than about the moon or Mars, says Jon Copley. (Image credit: Giordano Cipriani via Getty Images)

SP: Research has made great strides in recent years, I'm sure. What are some of the most exciting, new discoveries you discuss in the book?

JC: We're finding out a lot more about how deep sea animals interact with each other and their environment. An area where we've seen a lot of interesting papers
over the past five years has been in sensory ecology — realizing how animals perceive their environment, how they respond to that, how they avoid being seen by
predators... It was nice to bring some of those together in a couple of the chapters.
</item 90>

<item 91>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
The thing we’re jacked into and collectively creating seems more like a global endocrine system than anything we might have visualized in the years while
“cyber” was a prefix. This may seem a banal observation, given that Marshall McLuhan was talking about the global nervous system more than fifty years ago. We
had enthusiasm about cybernetics and global connectivity over the decades and, more recently, a revitalization of theory about networks and kinship and rhizomes
and all the rest. (The irony is that with fifty years of talk on “systems thinking,” we still have responses to things like COVID-19 or climate that are almost
antithetical to considering interconnected systems — dominated by one set of expertise and failing to incorporate the social sciences and humanities). So —
globally connected, yet divided into silos, camps, echo-chambers, and so on. Social media platforms are acting as agents, structuring our interactions and our
spaces for dialogue and solution-building.
</item 91>

<item 92>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
Importantly, the measurement and magnitude of the observed pattern effect are still debated and uncertain. The current best estimate is that the radiative
feedback or damping under idealized, long-term carbon dioxide forcing is 0.5 ± 0.5 watt per square meter per kelvin weaker than the radiative feedback seen
since the late 1800s [Andrews et al., 2022]; 0.5 watt per square meter per kelvin is similar in magnitude to the single radiative feedbacks that the pattern
effect modifies. The wide range of uncertainty opens two contrasting possibilities: The pattern effect might have acted to retard global warming fairly
strongly—in other words, mean global warming might have been much higher by now if the recent surface warming had come about in a different spatial pattern.
This possibility implies that the pattern effect has the potential to influence future near-term warming rates strongly as the surface warming pattern evolves.
</item 92>

<item 93>
Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' (Fortune,
2023-10-10T20:17:09+00:00)
Roughly 60% of Gen Zers say they are stressed about money this year more than last year, according to a Bankrate survey from July. It’s no surprise either—85%
of Gen Zers say that they couldn’t afford one month’s expenses if they lost their job today.

And since young people are typically affected by inflation the most, as they are the most likely to work part-time or low-paying jobs, this economic climate may
have left Gen Z with permanent “psychological scars,” one expert says.

“How can young people build careers or wealth if they don’t have jobs, and prices of goods and services continue to increase?” Dayo Abinusawa, founder of
London’s Awa Business School and a former lecturer at Cambridge University’s Judge Business School, previously told Fortune.

One Fidelity survey backs this argument: 45% of 18- to 35-year-olds “don’t see a point in saving until things return to normal.”
</item 93>

<item 94>
'We know far more about the deep ocean than the moon or Mars,' says explorer Jon Copley (Live Science: The Most Interesting Articles, 2023-09-28T11:01:16+00:00)
We can go out at night and if we look at the sky, we might wonder about what's going on up there. But you can't glance into the deep sea in the same way, so it
has become a realm of myth and darkness. Even the names of the deepest bits of the deep sea — the abyssal plains and the hadal zone — evoke that kind of
underworld. It's nice to be able to shine a light on that for people and to highlight how our lives are connected to it.

The deep sea and the names we give some its features evoke "some kind of underworld," says Jon Copley. (Image credit: A. Martin UW Photography via Getty Images)

SP: Speaking of the sky, how does exploring the deep ocean inform the search for life outside our solar system ?

JC: Deep sea exploration has shown us that the range of conditions under which life can thrive is far greater than we imagined.
</item 94>

<item 95>
Consistent policy is key to unlocking the climate tech boom, but the UK’s U-turn isn’t helping (TechCrunch, 2023-10-13T17:00:20+00:00)
Then came the nadir of the last few weeks.

Last week, with the government’s already infamous U-turn on its green pledges, the nation joined in a consternated chorus with global leaders to lament the
prime minister’s short-sighted choice. Sunak has pushed back the British net-zero transition timeline by at least five years.

The first and most galling concern is, obviously, the consequences for the future of our species on this planet. The next biggest issue, currently being voiced
by leaders across industries and especially within the climate tech and climate finance sectors, is the message it sends out to those of us in the trenches
actually trying to build technology to change the world and enable a sustainable future.

That message is loud and clear: The U.K. government isn’t willing to be consistent when it comes to climate crisis response policy, which, aside from capital
and the support of nascent tech markets, is one of the most critical things anyone in our sector can hope for.
</item 95>

<item 96>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
For example, the cooling pattern during the Last Glacial Maximum (LGM) differed from the warming pattern observed currently and from those expected for the next
couple of decades and expected in the equilibrium following a doubling of carbon dioxide, not only in sign but also in the spatial distribution of magnitudes.
The LGM currently provides the best constraint on the upper bound of equilibrium climate sensitivity [Sherwood et al., 2020], and hence, the details of the
pattern effect matter greatly.

Outstanding questions concern how representative pattern changes in the past century, the past millennium, or quasi-equilibrated times millions of years ago are
of expected future changes.
</item 96>

<item 97>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
Right now, the current plot is about the centralization of discourse on a few corporate platforms. The rise of the platforms brings potential to network
democratic uprisings, as well as buoy authoritarian leaders through post-truth memes and algorithms optimized to dish out anger and hatred. This is a more
challenging story to narrate, because the setting is everywhere. The story unfolds in our bedrooms while we should be sleeping or waking up, filling the most
quotidian moments of waiting in line in the grocery store or while in transit. The characters are us, even more intimately than with climate change. It makes it
hard to see the shape and meaning of this story. And while we are increasingly aware of the influence that shifting our media and social lives onto big tech
platforms has on our democracy, less attention is devoted to the influence this has on our ability to respond to climate change.

Think about these three forces meeting — climate change, authoritarianism, the Internet. What comes to mind?
</item 97>

<item 98>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
“Not as a NASA climate scientist. That’s really important for retaining my job.”)

He worries a third arrest could cost him.

“Do I keep doing science? Or do I, you know, keep engaging in risky activism and maybe lose that job?” Kalmus said.

Meanwhile, Kalmus has grown frustrated that more scientists aren’t agitating at protests and willing to face arrest. And his view of our climate predicament has
grown increasingly dark.

“It feels like it’s worse than I thought it would be in 2023,” Kalmus said, pointing to record-low Antarctic sea ice and record-high sea surface and land
temperatures as signs that Earth systems could be shifting more quickly than the scientific community can grasp.

In recent years, climate change has begun to sting Kalmus’ personal life.

In summer 2020, he sickened himself hiking in a California heat wave, then watched a wildfire burn a few miles from his California home, spewing smoke that left
his voice raspy and his head aching.
</item 98>

<item 99>
Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' (Fortune,
2023-10-10T20:17:09+00:00)
Some Gen Zers have even adopted the mentality that “money isn’t real” and are justifying spending on items to “treat themselves” amid a bleak reality.

Why it matters

Of course, every generation has lived through era-defining historical events. Millennials remember the 9/11 terrorist attacks and the U.S. invasion of Iraq.
Baby boomers lived through the civil rights movement and the Vietnam War. But no other generation has been plugged into the world through the internet from such
a young age like Gen Z.

Gen Z is already wielding their power in notable and sometimes comical ways. In June 2020, teenage TikTok users (with the help of K-pop fans) claimed to have
sunk a Trump campaign rally by registering for thousands of tickets with no intention of actually attending.
</item 99>

<item 100>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
Even worse, the system is addictive, with notifications delivering hits of dopamine in a part of what historian and addiction expert David Courtwright calls
“limbic capitalism.” Society has more or less sleepwalked into this outrage-industrial complex without having a real analytic framework for understanding it.
The tech platforms and some research groups or think tanks offer up “misinformation” or “disinformation” as the framework, which present the problem as if the
problem is bad content poisoning the well, rather than the structure itself being rotten. As Evgeny Morozov has quipped, “Post-truth is to digital capitalism
what pollution is to fossil capitalism — a by-product of operations.”
</item 100>
=== END ITEM id=99c62fb98b649851 ===

=== ITEM id=60a3faa163914f07 ===
Gold sentences and facts:
Sentence 0: Wang testified that he heard Bankman-Fried tell another trader that it was okay to continue withdrawing money from the account as long as the amount was less than FTX’s total trading revenue.
  Fact 0.0: Wang testified that he heard Bankman-Fried tell another trader that it was okay to continue withdrawing money from the account.
  Fact 0.1: Bankman-Fried said it was okay to keep withdrawing money as long as the amount was less than FTX's total trading revenue.
Sentence 1: Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front, Bankman-Fried had “secret access” to customer money—both cash and crypto—the government claimed.
  Fact 1.0: Caroline Ellison was Bankman-Fried's on-again off-again girlfriend.
  Fact 1.1: Caroline Ellison was the CEO of Alameda.
  Fact 1.2: Bankman-Fried used Caroline Ellison as a front.
  Fact 1.3: Through this front, Bankman-Fried had 'secret access' to customer money.
  Fact 1.4: The customer money included both cash and crypto.
  Fact 1.5: The government claimed that Bankman-Fried had this secret access to customer money.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 2.1: The defense countered that the FTX founder acted in good faith.
  Fact 2.2: The defense countered that the FTX founder never meant to commit fraud or steal.
  Fact 2.3: The defense countered that the FTX founder basically got in over his head.

Retrieved items:

=== END ITEM id=60a3faa163914f07 ===

=== ITEM id=0e902137adfa1b64 ===
Gold sentences and facts:
Sentence 0: "We all found ourselves at home looking at the people we worked with, the people we went out for drinks with and the people in our extended family on a computer screen," he told Fox News Digital.
  Fact 0.0: A speaker said that they all found themselves at home looking at the people they worked with, the people they went out for drinks with, and the people in their extended family on a computer screen.
  Fact 0.1: The speaker told Fox News Digital this.
Sentence 1: We used to play in person, but since the pandemic I run the game virtually online from my house.
  Fact 1.0: The speaker used to play the game in person.
  Fact 1.1: Since the pandemic, the speaker runs the game virtually online from their house.

Retrieved items:

=== END ITEM id=0e902137adfa1b64 ===

=== ITEM id=57718ef02311380e ===
Gold sentences and facts:
Sentence 0: When Google’s search head Prabhakar Raghavan testified in court on October 28, he revealed that the tech giant had paid $26.3 billion in 2021 to multiple browsers, phones and platforms, from companies including Apple, Samsung and Mozilla, The Verge reports.
  Fact 0.0: Google's search head Prabhakar Raghavan testified in court on October 28.
  Fact 0.1: Prabhakar Raghavan revealed in court testimony that Google paid $26.3 billion in 2021.
  Fact 0.2: The $26.3 billion was paid to multiple browsers, phones and platforms.
  Fact 0.3: The payments were made to companies including Apple, Samsung and Mozilla.
  Fact 0.4: The Verge reports this.
Sentence 1: It will attempt to argue that Google restrains competition within two separate markets, including the distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.
  Fact 1.0: The case (the sentence's subject, referred to as 'It') will attempt to argue that Google restrains competition within two separate markets.
  Fact 1.1: One of the two markets is the distribution of apps to Android users.
  Fact 1.2: The other market is the market for payment processing solutions for content inside Android apps.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content through anticompetitive means.
  Fact 2.2: The case argues that Google 'siphons off' news publishers' readers through anticompetitive means.
  Fact 2.3: The case argues that Google 'siphons off' news publishers' ad revenue through anticompetitive means.

Retrieved items:

=== END ITEM id=57718ef02311380e ===
